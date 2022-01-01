from tqdm import tqdm
import os
import json
import pandas as pd

def print_swag_overleaf_style_average_deltas(df):
    for ex_num in df['examples'].unique():
        print('\hline')
        for aug in df['aug'].unique():
            row = df[(df['aug']==aug) & (df['examples'] == ex_num)]
            latex_line = f"{row['examples'].values[0]} & \\verb|{row['aug'].values[0]}| & {row['accuracy'].values[0]}\\\\"
            print(latex_line)

def print_swag_overleaf_style(df):
    for dataset in df['dataset'].unique():
        print('\hline')
        for ex_num in df['examples'].unique():
            for aug in df['aug'].unique():
                row = df[(df['aug']==aug) & (df['dataset'] == dataset) & (df['examples'] == ex_num)]
                latex_line = f"\\verb|{row['dataset'].values[0]}| & {row['examples'].values[0]} & \\verb|{row['aug'].values[0]}| & {row['accuracy'].values[0]}\\\\"
                print(latex_line)

def get_swag_results_df(swag_results_path):
    df_all = pd.DataFrame()

    for exp in os.listdir(swag_results_path):
        exp_path = f'{swag_results_path}/{exp}'
        for num_examples in tqdm([16, 32, 64, 128, 256, 512, 1024], desc='Examples'):
            for seed in tqdm([42, 43, 44, 45, 46], desc='Seeds'):
                res_folder_path = f'{exp_path}/output-{num_examples}-{seed}'
                if 'eval_results.json' in os.listdir(res_folder_path):
                    res_file = f'{res_folder_path}/eval_results.json'
                    with open(res_file, "r") as f:
                        data = json.load(f)
                    res_dict = {'exp':exp, 'examples': num_examples, 'seed': seed, 'accuracy': data['eval_accuracy'],
                                                          'loss': data['eval_loss']}
                    df_all = df_all.append(res_dict, ignore_index=True)

    return df_all

def get_average_swag_res():
    swag_results_df = get_swag_results_df('swag_res')
    swag_results_dict = swag_results_df.to_dict()

    averages_df = pd.DataFrame()
    df = pd.DataFrame(swag_results_dict)
    for exp in df['exp'].unique():
        df_exp = df[df['exp'] == exp]
        for examples in df['examples'].unique():
            df_exp_examples = df_exp[df_exp['examples'] == examples]
            df_exp_examples_mean = df_exp_examples.mean(axis=0)
            df_exp_examples_mean['exp'] = exp
            df_exp_examples_mean['examples'] = examples
            averages_df = averages_df.append(df_exp_examples_mean, ignore_index=True)
    return averages_df

def get_average_over_seeds_df():
    averages_df = get_average_swag_res()
    print('averages_df')
    print(averages_df)
    averages_df = averages_df.round(3) # Average to 3rd decimal
    averages_df['aug'] =  ['_'.join(x.split("-")[1:]) for x in averages_df['exp']]
    averages_df['dataset'] = [x.split("-")[0] for x in averages_df['exp']]

    ### PRINT FULL AVERAGES OVER SEED TABLE
    print_swag_overleaf_style(averages_df)
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

def get_swag_deltas_over_average_df(averages_df):
    print('get_swag_deltas_over_average_df')
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
                          'accuracy': calc_diff(aug_df['accuracy'], baseline['accuracy'])}
            delta_df = delta_df.append(delta_dict, ignore_index=True)

    print(delta_df)
    return delta_df

def get_swag_deltas_df(averages_df):
    print('get_swag_deltas_df')
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
                              'accuracy': calc_diff(aug_df['accuracy'], baseline['accuracy'])}
                delta_df = delta_df.append(delta_dict, ignore_index=True)

    print(delta_df)
    return delta_df

if __name__ == '__main__':
    swag_results_path = 'swag_res'
    df_all = get_swag_results_df(swag_results_path)
    averages_df = get_average_over_seeds_df()
    # delta_df = get_swag_deltas_df(averages_df)
    # average_across_dataests_df = get_average_across_dataests(delta_df)

    avg_data_avg_seed_df = get_average_across_dataests(averages_df)
    avg_data_avg_seed_deltas_df = get_swag_deltas_over_average_df(avg_data_avg_seed_df)
    print("Full Res, Model:Roberta-base, Dataset: SWAG, Averaged over 5 seeds (42-46)")
    print(avg_data_avg_seed_deltas_df)
    print_swag_overleaf_style_average_deltas(avg_data_avg_seed_deltas_df)
    import pdb; pdb.set_trace()
    print('Done')
