import os
import numpy as np
list_params = ["num_to_start","cuda","sub_ts","size","well_rad",
               "output_values","output_images","image_quality","video_quality",
               "fps","cell_rad","velocity","noise_ratio",
               "initial_seed_ratio","cell_interaction_rad","cluster_threshold",
               "cluster_interaction_threshold","cluster_record_interval",
               "cluster_timer","u_bb","u_rb","u_yb","u_rr","alpha", "u_repulsion"]
test_params = ["dox_ratio","aba_ratio","u_yy","u_ry","replication_type", "end_step"]

def make_yaml(directory, default=True):
    dox_ratio = get_ratios(directory, "dox")
    aba_ratio = get_ratios(directory, "aba")
    u_yy = [40]
    u_ry = [10]
    replication_type = ['Default', 'None']
    os.chdir(directory + "/yaml_parameters")
    if default:
        with open("../template.yaml") as template_file:
            template_lines = template_file.readlines()
            for i in range(len(dox_ratio)):
                for yy_param in u_yy:
                    for ry_param in u_ry:
                        for rep_param in replication_type:
                            #with open(f"{i}dox_aba_{yy_param}yy_{ry_param}ry.yaml", "w") as yaml_file:
                            with open(f"{i}_dox_aba_{rep_param}_replication.yaml", "w") as yaml_file:
                                for line in template_lines:
                                    yaml_file.write(line)
                                yaml_file.write(f'\ndox_ratio: {dox_ratio[i]}\n')
                                yaml_file.write(f'aba_ratio: {aba_ratio[i]}\n')
                                yaml_file.write(f'u_yy: {yy_param}\n')
                                yaml_file.write(f'u_ry: {ry_param}\n')
                                yaml_file.write(f'replication_type: {rep_param}\n')
                                if rep_param == 'Default':
                                    yaml_file.write('end_step: 240')
                                if rep_param == 'None':
                                    yaml_file.write('end_step: 960')
                            yaml_file.close()
        template_file.close()

        print('done')
    else:
        print('not currently available')

def get_ratios(directory, cell_type):
    os.chdir(directory)
    ratio = np.loadtxt(f"channel_intensities_{cell_type}.csv", delimiter=',', encoding='utf-8-sig')
    return ratio

if __name__ == "__main__":
    make_yaml("/Users/andrew/PycharmProjects/CHO_adhesion_model")