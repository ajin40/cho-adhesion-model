import numpy as np
import random as r
import math
from numba import jit, prange
from pythonabm import Simulation, record_time, template_params
from pythonabm.backend import record_time, check_direct, template_params, check_existing, get_end_step, Graph, \
    progress_bar, starting_params, check_output_dir, assign_bins_jit, get_neighbors_cpu, get_neighbors_gpu


@jit(nopython=True, parallel=True)
def get_neighbor_forces(number_edges, edges, edge_forces, locations, center, types, radius, alpha=10, r_e=1.01,
                        u_bb=5, u_rb=1, u_yb=1, u_rr=20, u_ry=12, u_yy=30, u_repulsion=10000):
    sum_force = 0
    for index in range(number_edges):
        # get indices of cells in edge
        cell_1 = edges[index][0]
        cell_2 = edges[index][1]
        adhesion_values = np.reshape(np.array([u_bb, u_rb, u_yb, u_yb, u_rr, u_ry, u_rb, u_ry, u_yy]), (3, 3))
        # get cell positions
        cell_1_loc = locations[cell_1] - center
        cell_2_loc = locations[cell_2] - center

        # get new location position
        vec = cell_2_loc - cell_1_loc
        dist = np.linalg.norm(vec)

        # based on the distance apply force differently
        if dist == 0:
            edge_forces[index][0] = alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])
            edge_forces[index][1] = alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])
            sum_force += np.linalg.norm(edge_forces[index][0])
        elif 0 < dist < 2 * radius:
            edge_forces[index][0] = -1 * u_repulsion * (vec / dist)
            edge_forces[index][1] = 1 * u_repulsion * (vec / dist)
            sum_force -= np.linalg.norm(edge_forces[index][0])
        else:
            # get the cell type
            cell_1_type = types[cell_1]
            cell_2_type = types[cell_2]
            u = adhesion_values[cell_1_type, cell_2_type]
            # get value prior to applying type specific adhesion const
            value = (dist - r_e) * (vec / dist)
            edge_forces[index][0] = u * value + alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])
            edge_forces[index][1] = -1 * u * value + alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])
            sum_force += np.linalg.norm(edge_forces[index][0])
    return edge_forces, sum_force


@jit(nopython=True, parallel=True)
def get_gravity_forces(number_cells, locations, center, well_rad, net_forces):
    for index in range(number_cells):
        new_loc = locations[index] - center
        net_forces[index] = -1 * (new_loc / well_rad) * np.sqrt(1 - (np.linalg.norm(new_loc) / well_rad) ** 2)
    return net_forces


@jit(nopython=True)
def convert_edge_forces(number_edges, edges, edge_forces, neighbor_forces):
    for index in range(number_edges):
        # get indices of cells in edge
        cell_1 = edges[index][0]
        cell_2 = edges[index][1]

        neighbor_forces[cell_1] += edge_forces[index][0]
        neighbor_forces[cell_2] += edge_forces[index][1]

    return neighbor_forces


def set_div_thresh(cell_type):
    """ Specify division threshold value for a particular cell.

        Distribution of cell division thresholds modeled by a shifted gamma distribution
        from Stukalin et. al., RSIF 2013
    """
    # parameters for gamma distribution
    alpha, a_0, beta = 12.5, 10.4, 0.72

    # based on cell type return division threshold in seconds
    if cell_type > 0:
        alpha, a_0, beta = 12.5, 10.4, 0.72
        hours = r.gammavariate(alpha, beta) + a_0
        # CHO cell time < HEK cell time
    else:
        alpha, a_0, beta = 10, 10.4, 0.72
        hours = r.gammavariate(alpha, beta) + a_0

    return hours * 3600


def seed_cells(num_cells, radius, well_dimensions):
    # radius of the circle
    # center of sphere (x, y, z)
    center_x = well_dimensions[0] / 2
    center_y = well_dimensions[1] / 2
    center_z = well_dimensions[2] / 2
    locations = np.zeros((num_cells,3))
    # random angle
    if center_z > 0:
        for i in range(num_cells):
            phi = 2 * math.pi * r.random()
            theta = 2 * math.pi * r.random()
            rad = radius * math.sqrt(r.random())
            x = rad * math.cos(theta) * math.sin(phi) + center_x
            y = rad * math.sin(theta) * math.sin(phi) + center_y
            z = rad * math.cos(phi) + center_z
            locations[i] = np.array([x, y, z])
        return locations
    # random radius
    for i in range(num_cells):
        theta = 2 * math.pi * r.random()
        rad = radius * math.sqrt(r.random())
        x = rad * math.cos(theta) + center_x
        y = rad * math.sin(theta) + center_y
        locations[i] = np.array([x, y, 0])
    return locations


class TestSimulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self, yaml_file):
        # initialize the Simulation object
        Simulation.__init__(self)

        # read parameters from YAML file and add them to instance variables
        self.yaml_parameters(yaml_file)
        self.yaml_name = yaml_file
        # aba/dox/cho ratio
        self.cho_ratio = 1 - (self.aba_ratio + self.dox_ratio)
        self.aba_color = np.array([255, 255, 0], dtype=int) #yellow
        self.dox_color = np.array([255, 50, 50], dtype=int) #red
        self.cho_color = np.array([50, 50, 255], dtype=int)

        # movement parameters
        self.noise_magnitude = self.velocity * self.noise_ratio

        self.initial_seed_rad = self.well_rad * self.initial_seed_ratio
        self.dim = np.asarray(self.size)
        self.size = self.dim * self.well_rad

    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """
        # determine the number of agents for each cell type
        num_aba = int(self.num_to_start * self.aba_ratio)
        num_dox = int(self.num_to_start * self.dox_ratio)
        num_cho = int(self.num_to_start * self.cho_ratio)

        # add agents to the simulation
        self.add_agents(num_aba, agent_type="ABA")
        self.add_agents(num_dox, agent_type="DOX")
        self.add_agents(num_cho, agent_type="CHO")

        # indicate agent arrays and create the arrays with initial conditions
        self.indicate_arrays("locations", "radii", "colors", "cell_type", "division_set", "div_thresh")

        # generate random locations for cells
        self.locations = seed_cells(self.number_agents, self.initial_seed_rad, self.size)
        self.radii = self.agent_array(initial=lambda: self.cell_rad)

        # Define cell types, 2 is ABA, 1 is DOX, 0 is non-cadherin expressing cho cells
        self.cell_type = self.agent_array(dtype=int, initial={"ABA": lambda: 2, "DOX": lambda: 1, "CHO": lambda: 0})
        self.colors = self.agent_array(dtype=int, vector=3, initial={"ABA": lambda: self.aba_color, "DOX": lambda: self.dox_color, "CHO": lambda: self.cho_color})

        # setting division times (in seconds):
        self.div_thresh = self.agent_array(initial={"ABA": lambda: set_div_thresh(0), "DOX": lambda: set_div_thresh(0), "CHO": lambda: set_div_thresh(0)})
        self.division_set = self.agent_array(initial={"ABA": lambda: 17 * 3600 * r.random(), "DOX": lambda: 17 * 3600 * r.random(), "CHO": lambda: 16 * 3600 * r.random()})

        # save parameters to text file
        self.save_params(self.yaml_name)

        #indicate and create graphs for identifying neighbors
        self.indicate_graphs("neighbor_graph")
        self.neighbor_graph = self.agent_graph()

        #I want to keep track of the total of the attraction vs repulsion forces. See if there's a steady state
        self.track_forces = np.zeros(self.end_step)
        self.track_counter = 0

        # record initial values
        self.step_values()
        # self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # preform 60 subsets, each velocity / .05 seconds long
        for i in range(self.sub_ts):
            # get all neighbors within threshold (1.6 * diameter)
            self.get_neighbors(self.neighbor_graph, self.cell_interaction_rad * self.cell_rad)
            # increase division counter and determine if any cells are dividing
            self.reproduce(self.velocity/.05)

            # move the cells and track total repulsion vs adhesion forces
            self.move_parallel()
            self.noise(self.noise_magnitude)

            # add/remove agents from the simulation
            self.update_populations()
        # get the following data. We can generate images at each time step, but right now that is not needed.
        self.step_values()
        # self.step_image()
        self.track_counter += 1
        self.temp()
        # self.data()

    def end(self):
        """ Overrides the end() method from the Simulation class.
        """
        self.save_forces()
        self.step_values()
        self.step_image()

    @record_time
    def update_populations(self):
        """ Adds/removes agents to/from the simulation by adding/removing
            indices from the cell arrays and any graphs.
        """
        # get indices of hatching/dying agents with Boolean mask
        add_indices = np.arange(self.number_agents)[self.hatching]
        remove_indices = np.arange(self.number_agents)[self.removing]

        # count how many added/removed agents
        num_added = len(add_indices)
        num_removed = len(remove_indices)
        # go through each agent array name
        for name in self.array_names:
            # copy the indices of the agent array data for the hatching agents
            copies = self.__dict__[name][add_indices]

            # add indices to the arrays
            self.__dict__[name] = np.concatenate((self.__dict__[name], copies), axis=0)

            # if locations array
            if name == "locations":
                # go through the number of cells added
                for i in range(num_added):
                    # get mother and daughter indices
                    mother = add_indices[i]
                    daughter = self.number_agents + i

                    # move distance of radius in random direction
                    vec = self.radii[i] * self.random_vector()
                    self.__dict__[name][mother] += vec
                    self.__dict__[name][daughter] -= vec

            # reset division time
            if name == "division_set":
                # go through the number of cells added
                for i in range(num_added):
                    # get mother and daughter indices
                    mother = add_indices[i]
                    daughter = self.number_agents + i

                    # set division counter to zero
                    self.__dict__[name][mother] = 0
                    self.__dict__[name][daughter] = 0

            # set new division threshold
            if name == "division_threshold":
                # go through the number of cells added
                for i in range(num_added):
                    # get daughter index
                    daughter = self.number_agents + i

                    # set division threshold based on cell type
                    self.__dict__[name][daughter] = set_div_thresh(self.cell_type[daughter])

            # remove indices from the arrays
            self.__dict__[name] = np.delete(self.__dict__[name], remove_indices, axis=0)

        # go through each graph name
        for graph_name in self.graph_names:
            # add/remove vertices from the graph
            self.__dict__[graph_name].add_vertices(num_added)
            self.__dict__[graph_name].delete_vertices(remove_indices)

        # change total number of agents and print info to terminal
        self.number_agents += num_added
        # print("\tAdded " + str(num_added) + " agents")
        # print("\tRemoved " + str(num_removed) + " agents")

        # clear the hatching/removing arrays for the next step
        self.hatching[:] = False
        self.removing[:] = False

    @record_time
    def move_parallel(self):
        edges = np.asarray(self.neighbor_graph.get_edgelist())
        num_edges = len(edges)
        edge_forces = np.zeros((num_edges, 2, 3))
        center = self.size / 2
        neighbor_forces = np.zeros((self.number_agents, 3))
        # get adhesive/repulsive forces from neighbors and gravity forces
        edge_forces, adhesion_repulsion_force = get_neighbor_forces(num_edges, edges, edge_forces, self.locations, center, self.cell_type,
                                          self.cell_rad, u_bb=self.u_bb, u_rb=self.u_rb, u_rr=self.u_rr, u_yb=self.u_yb,
                                          u_ry=self.u_ry, u_yy=self.u_yy, alpha=self.alpha, u_repulsion=self.u_repulsion)
        neighbor_forces = convert_edge_forces(num_edges, edges, edge_forces, neighbor_forces)
        for i in range(self.number_agents):
            if np.linalg.norm(neighbor_forces[i]) != 0:
                neighbor_forces[i] = neighbor_forces[i] / np.linalg.norm(neighbor_forces[i])
            else:
                neighbor_forces[i] = 0
        # update locations based on forces
        self.locations += 2 * self.velocity * self.cell_rad * neighbor_forces
        # check that the new location is within the space, otherwise use boundary values
        self.locations = np.where(self.locations > self.well_rad, self.well_rad, self.locations)
        self.locations = np.where(self.locations < 0, 0, self.locations)
        self.track_forces[self.track_counter] += adhesion_repulsion_force/self.number_agents

    @record_time
    def reproduce(self, ts):
        """ If the agent meets criteria, hatch a new agent.
        """
        # increase division counter by time step for all agents
        self.division_set += ts

        #go through all agents marking for division if over the threshold
        if self.replication_type == 'Contact_Inhibition':
            adjacency_matrix = self.neighbor_graph.get_adjacency()
            for index in range(self.number_agents):
                if self.division_set[index] > self.div_thresh[index]:
                    # 12 is the maximum number of cells that can surround a cell
                    if np.sum(adjacency_matrix[index,:]) < 12:
                        self.mark_to_hatch(index)
        if self.replication_type == 'Default':
            for index in range(self.number_agents):
                if self.division_set[index] > self.div_thresh[index]:
                    self.mark_to_hatch(index)
        if self.replication_type == 'None':
            return

    @classmethod
    def simulation_mode_0(cls, name, output_dir, yaml_file="general.yaml"):
        """ Creates a new brand new simulation and runs it through
            all defined steps.
        """
        # make simulation instance, update name, and add paths
        sim = cls(yaml_file)
        sim.name = name
        sim.set_paths(output_dir)

        # set up the simulation agents and run the simulation
        sim.full_setup()
        sim.run_simulation()

    def noise(self, alpha):
        self.locations += alpha * 2 * self.cell_rad * np.random.normal(size=(self.number_agents, 3)) * self.dim

    def save_params(self, path):
        """ Add the instance variables to the Simulation object based
            on the keys and values from a YAML file.
        """
        # load the dictionary
        params = template_params(path)

        # iterate through the keys adding each instance variable
        with open(self.main_path + "parameters.txt", "w") as parameters:
            for key in list(params.keys()):
                parameters.write(f"{key}: {params[key]}\n")
        parameters.close()

    def save_forces(self):
        with open(self.main_path + "forces.txt", "w") as file:
            for i in range(len(self.track_forces)):
                file.write(f'{i}, {self.track_forces[i]/60}\n')
        file.close()

    @classmethod
    def start_sweep(cls, output_dir, yaml_file, name, mode):
        """ Configures/runs the model based on the specified
            simulation mode.
        """
        # check that the output directory exists and get the name/mode for the simulation
        output_dir = check_output_dir(output_dir)

        # new simulation
        if mode == 0:
            # first check that new simulation can be made and run that mode.
            # i = 0
            # while os.path.isdir(output_dir + name):
            #     name = name + f'_{i}'
            #     i += 1
            print(f'Starting {name} ...')
            name = check_existing(name, output_dir, new_simulation=True)
            cls.simulation_mode_0(name, output_dir, yaml_file=yaml_file)

        # existing simulation
        else:
            # check that previous simulation exists
            name = check_existing(name, output_dir, new_simulation=False)

            # call the corresponding mode
            if mode == 1:
                cls.simulation_mode_1(name, output_dir)    # continuation
            elif mode == 2:
                cls.simulation_mode_2(name, output_dir)    # images to video
            elif mode == 3:
                cls.simulation_mode_3(name, output_dir)    # archive simulation
            else:
                raise Exception("Mode does not exist!")


if __name__ == "__main__":
    TestSimulation.start("/Users/andrew/PycharmProjects/CHO_adhesion_model/outputs/")
    # TestSimulation.start("/Users/andrew/PycharmProjects/pace_outputs")
    #TestSimulation.start("C:\\Research\\Code\\Tordoff_model_outputs")
