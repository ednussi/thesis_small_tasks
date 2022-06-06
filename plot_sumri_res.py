from tqdm import tqdm
import os
import json
import pandas as pd

def print_sumri_overleaf_style_average_deltas(df):
    for ex_num in df['examples'].unique():
        print('\hline')
        for aug in sorted(df['aug'].unique()):
            row = df[(df['aug']==aug) & (df['examples'] == ex_num)]
            latex_line = f"{row['examples'].values[0]} & \\verb|{row['aug'].values[0]}| & {row['rouge1'].values[0]} & {row['rouge2'].values[0]} & {row['rougeL'].values[0]} & {row['rougeLsum'].values[0]}\\\\"
            print(latex_line)

def print_sumri_overleaf_style(df):
    for dataset in df['dataset'].unique():
        print('\hline')
        for ex_num in df['examples'].unique():
            for aug in sorted(df['aug'].unique()):
                row = df[(df['aug']==aug) & (df['dataset'] == dataset) & (df['examples'] == ex_num)]
                latex_line = f"\\verb|{row['dataset'].values[0]}| & {row['examples'].values[0]} & \\verb|{row['aug'].values[0]}| & {row['rouge1'].values[0]} & {row['rouge2'].values[0]} & {row['rougeL'].values[0]} & {row['rougeLsum'].values[0]}\\\\"
                print(latex_line)

def print_overleaf_style_mean_var_datasets(df):
    aug_name_dict = {'baseline':'baseline', 'baseline_double':'baseline_double', 'lorem_ipsum':'lorem-ipsum','mosaic':'mosaic-concat','mosaic_crop':'mosaic'}
    df['aug'] = [aug_name_dict['_'.join(x.split("-")[1:])] for x in df['exp']]
    df['dataset'] = [x.split("-")[0] for x in df['exp']]
    df = df.round(3)
    for d in df['dataset'].unique():
        print('\hline \hline \multicolumn{6}{c}{\\textbf{\\verb|' + d + '|}} \\\\')
        for ex_num in df['examples'].unique():
            print('\hline')



            for aug in sorted(df['aug'].unique()):
                row = df[(df['dataset']==d) & (df['aug']==aug) & (df['examples'] == ex_num)]
                baseline_row = df[(df['dataset']==d) & (df['aug']=='baseline') & (df['examples'] == ex_num)]

                latex_line0 = f"{row['examples'].values[0]} & \\verb|{row['aug'].values[0]}| &"
                latex_line1 = f" ${row['rouge1'].values[0]:.3f} \pm {row['rouge1_var'].values[0]:.3f}$ &"
                latex_line2 = f" ${row['rouge2'].values[0]:.3f} \pm {row['rouge2_var'].values[0]:.3f}$ &"
                latex_line3 = f" ${row['rougeL'].values[0]:.3f} \pm {row['rougeL_var'].values[0]:.3f}$ &"
                latex_line4 = f" ${row['rougeLsum'].values[0]:.3f} \pm {row['rougeLsum_var'].values[0]:.3f}$\\\\"

                latex_line = latex_line0 + latex_line1 + latex_line2 + latex_line3 + latex_line4
                print(latex_line)

def get_sumri_results_df(sumri_results_path):
    df_all = pd.DataFrame()

    for exp in os.listdir(sumri_results_path):
        exp_path = f'{sumri_results_path}/{exp}'
        for num_examples in tqdm([1000000], desc='Examples'):
            for seed in tqdm([42], desc='Seeds'):

                res_folder_path = f'{exp_path}/output-{num_examples}-{seed}'
                print(res_folder_path)
                # import pdb; pdb.set_trace()
                if 'eval_results.json' in os.listdir(res_folder_path):
                    res_file = f'{res_folder_path}/eval_results.json'
                    with open(res_file, "r") as f:
                        data = json.load(f)
                    res_dict = {'exp':exp, 'examples': num_examples, 'seed': seed, 'rouge1': data['eval_rouge1'], 'rouge2': data['eval_rouge2'],
                                                          'rougeL': data['eval_rougeL'], 'rougeLsum': data['eval_rougeLsum'],
                                                          'loss': data['eval_loss']}
                    df_all = df_all.append(res_dict, ignore_index=True)
    return df_all

def get_average_sumri_res():
    averages_df = pd.DataFrame()
    df = get_sumri_results_df('sumri_res_full')
    for exp in df['exp'].unique():
        df_exp = df[df['exp'] == exp]
        for examples in df['examples'].unique():

            df_exp_examples = df_exp[df_exp['examples'] == examples]
            df_exp_examples_mean = df_exp_examples.mean(axis=0)
            df_exp_examples_mean['exp'] = exp
            df_exp_examples_mean['examples'] = examples


            df_exp_examples_var = df_exp_examples.std(axis=0)
            naming_dict = {x:x+'_var' for x in df_exp_examples_var.keys()}
            df_exp_examples_var = df_exp_examples_var.rename(naming_dict)
            df_exp_examples_mean['exp'] = exp
            df_exp_examples_mean['examples'] = examples

            df_mean_var = pd.concat([df_exp_examples_mean, df_exp_examples_var], axis=0)
            averages_df = averages_df.append(df_mean_var, ignore_index=True)
    return averages_df

def get_average_over_seeds_df():
    averages_df = get_average_sumri_res()
    print('averages_df')
    print(averages_df)
    averages_df = averages_df.round(3) # Average to 3rd decimal
    averages_df['aug'] =  [x.split("-")[1] for x in averages_df['exp']]
    averages_df['dataset'] = [x.split("-")[0] for x in averages_df['exp']]

    ### PRINT FULL AVERAGES OVER SEED TABLE
    print_sumri_overleaf_style(averages_df)
    print('FULL AVERAGES OVER SEED TABLE')
    return averages_df

def get_average_across_dataests(df):
    averages_df = pd.DataFrame()
    for aug in df['aug'].unique():
        for examples in df['examples'].unique():
            aug_examples_df = df[(df['examples'] == examples) & (df['aug'] == aug)]
            df_exp_examples_mean = aug_examples_df.mean(axis=0)
            df_exp_examples_mean['aug'] = aug
            averages_df = averages_df.append(df_exp_examples_mean, ignore_index=True)
    averages_df = averages_df.round(3)
    return averages_df

def calc_diff(v_base, v_new):
    diff_val = round(float(v_new) - float(v_base), 3)
    relative_diff_val = round(diff_val * 100 / float(v_base), 1)
    if diff_val >= 0:
        diff_return_str = str(f'+{diff_val}(+{relative_diff_val}\\%)')
    else:
        diff_return_str = str(f'{diff_val}({relative_diff_val}\\%)')
    return diff_return_str

def get_sumri_deltas_over_average_df(averages_df):
    print('get_sumri_deltas_over_average_df')
    delta_df = pd.DataFrame() #how much better / wrost is mosaic to baseline
    for examples in averages_df['examples'].unique():
        baseline = averages_df[(averages_df['aug']=='baseline') & (averages_df['examples'] == examples)]
        if baseline.empty:
            print(f'Skipping - examples {examples}')
            import pdb; pdb.set_trace()
            continue

        for aug in set(averages_df['aug'].unique()) - set(['baseline']):
            aug_df = averages_df[(averages_df['aug'] == aug) & (averages_df['examples'] == examples)]
            if aug_df.empty:
                print(f'Skipping - examples {examples}, aug {aug}')
                import pdb; pdb.set_trace()
                continue
            delta_dict = {'examples': examples, 'aug': aug,
                          'rouge1': calc_diff(baseline['rouge1'], aug_df['rouge1']),
                          'rouge2': calc_diff(baseline['rouge2'], aug_df['rouge2']),
                          'rougeL': calc_diff(baseline['rougeL'], aug_df['rougeL']),
                          'rougeLsum': calc_diff(baseline['rougeLsum'], aug_df['rougeLsum'])}
            delta_df = delta_df.append(delta_dict, ignore_index=True)

    print(delta_df)
    return delta_df

def get_sumri_deltas_df(averages_df):
    print('get_sumri_deltas_df')
    delta_df = pd.DataFrame() #how much better / wrost is mosaic to baseline
    for dataset in averages_df['dataset'].unique():
        for examples in averages_df['examples'].unique():
            baseline = averages_df[(averages_df['aug']=='baseline') & (averages_df['dataset'] == dataset) & (averages_df['examples'] == examples)]
            if baseline.empty:
                print(f'Skipping - empty baseline for dataset {dataset}, examples {examples}')
                import pdb; pdb.set_trace()
                continue

            for aug in set(averages_df['aug'].unique()) - set(['baseline']):
                aug_df = averages_df[(averages_df['aug'] == aug) & (averages_df['dataset'] == dataset) & (averages_df['examples'] == examples)]
                if aug_df.empty:
                    print(f'Skipping - empty aug_df for dataset {dataset}, examples {examples}, aug {aug}')
                    import pdb; pdb.set_trace()
                    continue

                delta_dict = {"dataset":dataset, 'examples': examples, 'aug': aug,
                              'rouge1': calc_diff(baseline['rouge1'], aug_df['rouge1']),
                              'rouge2': calc_diff(baseline['rouge2'], aug_df['rouge2']),
                              'rougeL': calc_diff(baseline['rougeL'], aug_df['rougeL']),
                              'rougeLsum': calc_diff(baseline['rougeLsum'], aug_df['rougeLsum'])}
                delta_df = delta_df.append(delta_dict, ignore_index=True)

    print(delta_df)
    return delta_df

if __name__ == '__main__':
    sumri_results_path = 'sumri_res_full'
    # df_all = get_sumri_results_df(sumri_results_path)
    averages_df = get_average_over_seeds_df()
    print_overleaf_style_mean_var_datasets(averages_df)
    import pdb; pdb.set_trace()

    averages_df = get_average_over_seeds_df()
    # delta_df = get_sumri_deltas_df(averages_df)
    # average_across_dataests_df = get_average_across_dataests(delta_df)

    avg_data_avg_seed_df = get_average_across_dataests(averages_df)
    avg_data_avg_seed_deltas_df = get_sumri_deltas_over_average_df(avg_data_avg_seed_df)
    print("Full Res, Model:55-small, prompt-summarize:, Averaged over 5 seeds (42-46)")
    print(avg_data_avg_seed_deltas_df)
    print_sumri_overleaf_style_average_deltas(avg_data_avg_seed_deltas_df)
    import pdb; pdb.set_trace()
    print('Done')
