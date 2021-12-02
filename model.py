import numpy as np
import random as r
import math
from numba import jit, prange
from pythonabm import Simulation, record_time, template_params
import cv2
from numba import cuda
import numba
from pythonabm.backend import record_time, check_direct, template_params, check_existing, get_end_step, Graph, \
    progress_bar, starting_params, check_output_dir, assign_bins_jit, get_neighbors_cpu, get_neighbors_gpu


@jit(nopython=True, parallel=True)
def get_neighbor_forces(number_edges, edges, edge_forces, locations, center, types, radius, alpha=10, r_e=1.01,
                        u_bb=5, u_rb=1, u_yb=1, u_rr=20, u_ry=12, u_yy=30):
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
        elif 0 < dist < 2 * radius:
            edge_forces[index][0] = -1 * (10 ** 4) * (vec / dist)
            edge_forces[index][1] = 1 * (10 ** 4) * (vec / dist)
        else:
            # get the cell type
            cell_1_type = types[cell_1]
            cell_2_type = types[cell_2]
            u = adhesion_values[cell_1_type, cell_2_type]
            # get value prior to applying type specific adhesion const
            value = (dist - r_e) * (vec / dist)
            edge_forces[index][0] = u * value + alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])
            edge_forces[index][1] = -1 * u * value + alpha * (2 * np.random.rand(3) - 1) * np.array([1, 1, 0])
    return edge_forces


@jit(nopython=True, parallel=True)
def get_gravity_forces(number_cells, locations, center, well_rad, net_forces):
    for index in range(number_cells):
        new_loc = locations[index] - center
        # net_forces[index] = -1 * (new_loc / well_rad) * np.sqrt((np.linalg.norm(new_loc) / well_rad) ** 2)
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
        from Stukalin et al., RSIF 2013
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

        # 1 is HEK293FT Cell (yellow), 0 is CHO K1 Cell (blue)
        self.cell_type = self.agent_array(dtype=int, initial={"ABA": lambda: 2, "DOX": lambda: 1, "CHO": lambda: 0})
        self.colors = self.agent_array(dtype=int, vector=3, initial={"ABA": lambda: self.aba_color, "DOX": lambda: self.dox_color, "CHO": lambda: self.cho_color})

        # setting division times (in seconds):
        self.div_thresh = self.agent_array(initial={"ABA": lambda: set_div_thresh(2), "DOX": lambda: set_div_thresh(1), "CHO": lambda: set_div_thresh(0)})
        self.division_set = self.agent_array(initial={"ABA": lambda: 17 * 3600 * r.random(), "DOX": lambda: 17 * 3600 * r.random(), "CHO": lambda: 16 * 3600 * r.random()})

        # indicate agent graphs and create the graphs for holding agent neighbors
        self.indicate_graphs("neighbor_graph", "cluster_graph")
        self.neighbor_graph = self.agent_graph()
        self.cluster_graph = self.agent_graph()
        self.save_params(self.yaml_name)
        # record initial values
        self.step_values()
        # self.get_clusters(self.cluster_threshold, self.cluster_record_interval, self.cluster_interaction_threshold)
        self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # preform 60 subsets, each velocity / .05 seconds long
        self.cluster_timer += 1
        for i in range(self.sub_ts):
            # increase division counter and determine if any cells are dividing
            self.reproduce(self.velocity/.05)

            # get all neighbors within threshold (1.6 * diameter)
            self.get_neighbors(self.neighbor_graph, self.cell_interaction_rad * self.cell_rad)

            # self.get_clusters(self.cluster_threshold, self.cluster_record_interval,
            #                   self.cluster_interaction_threshold)
            # move the cells
            self.move_parallel()
            self.noise(self.noise_magnitude)
            # add/remove agents from the simulation
            self.update_populations()
        # get the following data
        self.step_values()
        self.step_image()
        self.temp()
        self.data()

    def end(self):
        """ Overrides the end() method from the Simulation class.
        """
        self.create_video()

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

    def move_parallel(self):
        edges = np.asarray(self.neighbor_graph.get_edgelist())
        num_edges = len(edges)
        edge_forces = np.zeros((num_edges, 2, 3))
        center = self.size / 2
        neighbor_forces = np.zeros((self.number_agents, 3))
        grav_forces = np.zeros((self.number_agents, 3))
        total_force = np.zeros((self.number_agents, 3))

        # get adhesive/repulsive forces from neighbors and gravity forces
        edge_forces = get_neighbor_forces(num_edges, edges, edge_forces, self.locations, center, self.cell_type,
                                          self.cell_rad, u_bb=self.u_bb, u_rb=self.u_rb, u_rr=self.u_rr, u_yb=self.u_yb,
                                          u_ry=self.u_ry, u_yy=self.u_yy, alpha=self.alpha)
        neighbor_forces = convert_edge_forces(num_edges, edges, edge_forces, neighbor_forces)
        grav_forces = get_gravity_forces(self.number_agents, self.locations, center, self.well_rad, grav_forces)
        total_force = neighbor_forces # + grav_forces
        for i in range(self.number_agents):
            if np.linalg.norm(neighbor_forces[i]) != 0:
                total_force[i] = total_force[i] / np.linalg.norm(total_force[i])
            else:
                total_force[i] = 0
            # total_force[i] = total_force[i] / np.linalg.norm(total_force[i])
        # update locations based on forces
        self.locations += 2 * self.velocity * self.cell_rad * total_force
        # check that the new location is within the space, otherwise use boundary values
        self.locations = np.where(self.locations > self.well_rad, self.well_rad, self.locations)
        self.locations = np.where(self.locations < 0, 0, self.locations)

    @record_time
    def reproduce(self, ts):
        """ If the agent meets criteria, hatch a new agent.
        """
        # increase division counter by time step for all agents
        self.division_set += ts

        # go through all agents marking for division if over the threshold
        for index in range(self.number_agents):
            if self.division_set[index] > self.div_thresh[index]:
                self.mark_to_hatch(index)

    # Not used
    def remove_overlap(self, index):
        self.get_neighbors(self.neighbor_graph, 2*self.radii[index])
        while len(self.neighbor_graph.neighbors(index)) > 0:
            for neighbor_cell in self.neighbor_graph.neighbors(index):
                mag = np.linalg.norm(self.locations[neighbor_cell] - self.locations[index])
                vec = mag * np.random.rand(3) * self.dim
                self.locations[index] += vec
                self.locations[neighbor_cell] -= vec
            self.get_neighbors(self.neighbor_graph, 2*self.radii[index])

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

    def get_clusters(self, cluster_threshold, time_thresh, cluster_distance):
        # Create graphs of specified distance.
        if self.cluster_timer % time_thresh == 0:
            self.get_neighbors_clusters(self.cluster_graph, cluster_distance * self.cell_rad)
            # Identify unique clusters in graph
            clusters = self.cluster_graph.clusters()
            file_name = f"{self.name}_values_{self.current_step}_clusters.csv"
            cluster_file = open(self.values_path + file_name, "w")
            if len(clusters) > 0:
                centroids = np.zeros([len(clusters),3])
                radius = np.zeros(len(clusters))
                # Calculate Mean
                for i in range(len(clusters)):
                    if len(clusters[i]) > cluster_threshold:
                        location_graph = self.locations[clusters[i]]
                        centroids[i] = np.mean(location_graph,0)
                        max_distance = 0
                        # and Radius for circles
                        for j in range(len(clusters[i])):
                            if np.linalg.norm(location_graph[j]-centroids[i]) > max_distance:
                                max_distance = np.linalg.norm(location_graph[j]-centroids[i])
                        radius[i] = max_distance
                        cluster_file.write(f"{centroids[i][0]}, {centroids[i][1]}, {centroids[i][2]}, {radius[i]}\n")
                if len(centroids > 0):
                    self.step_image_cluster(centroids, radius)
            cluster_file.close()


    @record_time
    def step_image_cluster(self, centroids, radius, background=(0, 0, 0), origin_bottom=True):
        """ Creates an image of the simulation space with a cluster overlay.
        """
        # only continue if outputting images
        if self.output_images:
            # get path and make sure directory exists


            # get the size of the array used for imaging in addition to the scaling factor
            x_size = self.image_quality
            scale = x_size / self.size[0]
            y_size = math.ceil(scale * self.size[1])

            # create the agent space background image and apply background color
            image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
            background = (background[2], background[1], background[0])
            image[:, :] = background

            # go through all of the agents
            for index in range(self.number_agents):
                # get xy coordinates, the axis lengths, and color of agent
                x, y = int(scale * self.locations[index][0]), int(scale * self.locations[index][1])
                major, minor = int(scale * self.radii[index]), int(scale * self.radii[index])
                color = (int(self.colors[index][2]), int(self.colors[index][1]), int(self.colors[index][0]))

                # draw the agent and a black outline to distinguish overlapping agents
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, (0, 0, 0), 1)
            for i in range(len(centroids)):
                x = int(scale * centroids[i, 0])
                y = int(scale * centroids[i, 1])
                rad = int(radius[i] * scale)
                image = cv2.ellipse(image, (x, y), (rad, rad), 0, 0, 360, (0, 0, 255), 3)
            # if the origin should be bottom-left flip it, otherwise it will be top-left
            if origin_bottom:
                image = cv2.flip(image, 0)

            # save the image as a PNG
            image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)
            file_name = f"{self.name}_image_{self.current_step}_cluster.png"
            cv2.imwrite(self.images_path + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])

    @record_time
    def get_neighbors_clusters(self, graph, distance, clear=True):
        """ Finds all neighbors, within fixed radius, for each each agent.
        """
        # get graph object reference and if desired, remove all existing edges in the graph
        if clear:
            graph.delete_edges(None)

        # don't proceed if no agents present
        if np.sum(self.cell_type) == 0:
            return

        # assign each of the agents to bins, updating the max agents in a bin (if necessary)
        bins, bins_help, bin_locations, graph.max_agents = self.assign_bins(graph.max_agents, distance)

        # run until all edges are accounted for
        while True:
            # get the total amount of edges able to be stored and make the following arrays
            # We are only looking at HEK cells here
            length = np.sum(self.cell_type) * graph.max_neighbors
            edges = np.zeros((length, 2), dtype=int)         # hold all edges
            if_edge = np.zeros(length, dtype=bool)                 # say if each edge exists
            edge_count = np.zeros(np.sum(self.cell_type), dtype=int)   # hold count of edges per agent

            # if using CUDA GPU
            if self.cuda:
                # allow the following arrays to be passed to the GPU
                edges = cuda.to_device(edges)
                if_edge = cuda.to_device(if_edge)
                edge_count = cuda.to_device(edge_count)

                # specify threads-per-block and blocks-per-grid values
                tpb = 72
                bpg = math.ceil(self.number_agents / tpb)

                # call the CUDA kernel, sending arrays to GPU
                get_neighbors_gpu[bpg, tpb](cuda.to_device(self.locations), cuda.to_device(bin_locations),
                                            cuda.to_device(bins), cuda.to_device(bins_help), distance, edges, if_edge,
                                            edge_count, graph.max_neighbors)

                # return the following arrays back from the GPU
                edges = edges.copy_to_host()
                if_edge = if_edge.copy_to_host()
                edge_count = edge_count.copy_to_host()

            # otherwise use parallelized JIT function
            else:
                edges, if_edge, edge_count = get_neighbors_cpu(np.sum(self.cell_type),  self.locations[self.cell_type==1], bin_locations, bins,
                                                               bins_help, distance, edges, if_edge, edge_count,
                                                               graph.max_neighbors)

            # break the loop if all neighbors were accounted for or revalue the maximum number of neighbors
            max_neighbors = np.amax(edge_count)
            if graph.max_neighbors >= max_neighbors:
                break
            else:
                graph.max_neighbors = max_neighbors * 2

        # reduce the edges to edges that actually exist and add those edges to graph
        graph.add_edges(edges[if_edge])

        # simplify the graph's edges if not clearing the graph at the start
        if not clear:
            graph.simplify()

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

    @classmethod
    def start_sweep(cls, output_dir, yaml_file, name, mode):
        """ Configures/runs the model based on the specified
            simulation mode.
        """
        # check that the output directory exists and get the name/mode for the simulation
        output_dir = check_output_dir(output_dir)

        # new simulation
        if mode == 0:
            # first check that new simulation can be made and run that mode
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
    #TestSimulation.start("C:\\Research\\Code\\Tordoff_model_outputs")
