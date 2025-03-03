import matplotlib.pyplot as plt

all_results = [
    {
        'dataset_name': 'gowalla',
        'hyper_param_name': 'α',
        'hyper_param_values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'results': [
            {'MRR': 0.2907365697103148, 'Acc@1': 0.1987457082507605, 'Acc@5': 0.3923644242191939,
             'Acc@10': 0.47439630073642747},
            {'MRR': 0.28863610180653954, 'Acc@1': 0.19599192077356148, 'Acc@5': 0.3904424817252356,
             'Acc@10': 0.4734856207711149},
            {'MRR': 0.29111139708087314, 'Acc@1': 0.19782415449481727, 'Acc@5': 0.39267432725216594,
             'Acc@10': 0.47640795200308816},
            {'MRR': 0.2912316872594246, 'Acc@1': 0.1969705619303153, 'Acc@5': 0.3929570458085615,
             'Acc@10': 0.47816406918992976},
            {'MRR': 0.2899912684770961, 'Acc@1': 0.1952851243825726, 'Acc@5': 0.39158422974144846,
             'Acc@10': 0.4776774670592105},
            {'MRR': 0.288085706623746, 'Acc@1': 0.1934827935855509, 'Acc@5': 0.3890261704956002,
             'Acc@10': 0.4747279513506607},
            {'MRR': 0.2847285931525115, 'Acc@1': 0.19084861780528847, 'Acc@5': 0.3851904408506566,
             'Acc@10': 0.4701473670475212},
            {'MRR': 0.28066527122345686, 'Acc@1': 0.1883992964657462, 'Acc@5': 0.37881839954112606,
             'Acc@10': 0.4634545489143879},
            {'MRR': 0.27498986074971526, 'Acc@1': 0.18459075129737915, 'Acc@5': 0.3715846103241205,
             'Acc@10': 0.45488600189747647},
            {'MRR': 0.26898250418412034, 'Acc@1': 0.18189948811630607, 'Acc@5': 0.3614719850376641,
             'Acc@10': 0.44227240476598245},
            {'MRR': 0.26408837091810955, 'Acc@1': 0.1813965752996409, 'Acc@5': 0.3547492639802967,
             'Acc@10': 0.4274541465841346}
        ]
    },

    {
        'dataset_name': 'gowalla',
        'hyper_param_name': 'β',
        'hyper_param_values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'results': [
            {'MRR': 0.23242569000820923, 'MR': 2968.951660563752, 'Acc@1': 0.15939617840628287,
             'Acc@5': 0.30756788643413063, 'Acc@10': 0.3763391752773496},
            {'MRR': 0.2769335259930939, 'MR': 1952.5202755418545, 'Acc@1': 0.18555580021584475,
             'Acc@5': 0.3749228640477142,
             'Acc@10': 0.4580992070288183},
            {'MRR': 0.2908607952563962, 'MR': 2003.2216554258855, 'Acc@1': 0.20069211677363757,
             'Acc@5': 0.38812364587326054, 'Acc@10': 0.4718029016710298},
            {'MRR': 0.2947967585360145, 'MR': 2025.7962088528966, 'Acc@1': 0.2035872635290344,
             'Acc@5': 0.393565978083875, 'Acc@10': 0.4774572727989409},
            {'MRR': 0.2944312310116055, 'MR': 2078.836257023789, 'Acc@1': 0.20190726287660693,
             'Acc@5': 0.39429724050378273,
             'Acc@10': 0.4785663994432619},
            {'MRR': 0.29155376048923287, 'MR': 2067.4518060007013, 'Acc@1': 0.19690803763418938,
             'Acc@5': 0.39443588133432284, 'Acc@10': 0.47941455511244857},
            {'MRR': 0.28790236106268907, 'MR': 2048.2061262936413, 'Acc@1': 0.192194249395825,
             'Acc@5': 0.39189141432676283,
             'Acc@10': 0.4780281468070473},
            {'MRR': 0.2829864019870478, 'MR': 2031.543862152956, 'Acc@1': 0.1870455095322367,
             'Acc@5': 0.3871749076407408,
             'Acc@10': 0.473724844164988},
            {'MRR': 0.27842268634551304, 'MR': 2016.3542042152249, 'Acc@1': 0.18374803252350777,
             'Acc@5': 0.3812785946712989, 'Acc@10': 0.46869299755067867},
            {'MRR': 0.2741906794067343, 'MR': 2012.2238057179827, 'Acc@1': 0.18086104111108392,
             'Acc@5': 0.3729954846584406,
             'Acc@10': 0.45994775143601996},
            {'MRR': 0.2078347909672816, 'MR': 21463.833200129397, 'Acc@1': 0.13106179847060134,
             'Acc@5': 0.299768660104334,
             'Acc@10': 0.3740719899308699}
        ]
    },

    {
        'dataset_name': 'gowalla',
        'hyper_param_name': 'τ',
        'hyper_param_values': [6, 8, 12, 24],
        'results': [
            {'MRR': 0.2895736504599369, 'MR': 1956.5237605917516, 'Acc@1': 0.1979627953253574,
             'Acc@5': 0.3890479180768614, 'Acc@10': 0.47356173730552903},
            {'MRR': 0.2930262005090718, 'MR': 1958.7998135144908, 'Acc@1': 0.20182842789453512,
             'Acc@5': 0.39176636573451096, 'Acc@10': 0.4759539712442607},
            {'MRR': 0.29525100550912803, 'MR': 2019.8585659644916, 'Acc@1': 0.20423969096687028,
             'Acc@5': 0.39332131779468654, 'Acc@10': 0.47770736998344465},
            {'MRR': 0.29350035382457496, 'MR': 2187.2605944701336, 'Acc@1': 0.2022334765955249,
             'Acc@5': 0.3924595698872116, 'Acc@10': 0.47815863229461447}
        ]
    },

    {
        'dataset_name': 'gowalla',
        'hyper_param_name': 'λ',
        'hyper_param_values': [2, 4, 6, 8, 10],
        'results': []
    },
]


def print_hyper_param_analysis_single_figure(dataset_name, hyper_param_name, hyper_param_values, results):
    mrr_values = [res['MRR'] for res in results]
    acc1_values = [res['Acc@1'] for res in results]
    acc5_values = [res['Acc@5'] for res in results]
    acc10_values = [res['Acc@10'] for res in results]
    plt.title(dataset_name)
    plt.xlabel(f'Hyperparameter({hyper_param_name})')
    plt.ylabel('Results of Metrics')
    ticks = range(0, len(hyper_param_values), 1)

    plt.plot(ticks, mrr_values, label='MRR', color='red', marker='^')
    plt.plot(ticks, acc1_values, label='Acc@1', color='blue', marker='o')
    plt.plot(ticks, acc5_values, label='Acc@5', color='green', marker='s')
    plt.plot(ticks, acc10_values, label='Acc@10', color='orange', marker='D')
    plt.legend(loc='lower right')
    # 设置x轴的标签为你的超参数值的字符串表示
    plt.xticks(ticks=range(0, len(hyper_param_values), 1), labels=[str(x) for x in hyper_param_values])
    plt.grid()

    plt.savefig(f'Hyperparameter({hyper_param_name}) Analysis on {dataset_name}.png')
    plt.show()


def get_hyper_param_analysis_single_figure(axs, all_data, i):
    data = all_data[i]
    dataset_name, hyper_param_name, hyper_param_values, results = data['dataset_name'], data['hyper_param_name'], data[
        'hyper_param_values'], data['results']
    mrr_values = [res['MRR'] for res in results]
    acc1_values = [res['Acc@1'] for res in results]
    acc5_values = [res['Acc@5'] for res in results]
    acc10_values = [res['Acc@10'] for res in results]
    axs.set_xlabel(f"({chr(i + ord('a'))}) Hyperparameter({hyper_param_name})")

    ticks = range(0, len(hyper_param_values), 1)
    if len(hyper_param_values) >= 8:
        labels = [f'{x}' if i % 2 == 0 else '' for i, x in enumerate(hyper_param_values)]
    else:
        labels = [f'{x}' for x in hyper_param_values]

    axs.plot(ticks, mrr_values, label='MRR', color='red', marker='^')
    axs.plot(ticks, acc1_values, label='Acc@1', color='blue', marker='o')
    axs.plot(ticks, acc5_values, label='Acc@5', color='green', marker='s')
    axs.plot(ticks, acc10_values, label='Acc@10', color='orange', marker='D')
    # 设置x轴的标签为你的超参数值的字符串表示
    axs.set_xticks(ticks=ticks, labels=labels)
    axs.grid()
    return axs


def print_hyper_param_analysis_multi_figure(dataset_name):
    fig, axs = plt.subplots(2, 2, sharey='all')
    get_hyper_param_analysis_single_figure(axs[0, 0], all_results, 0)
    get_hyper_param_analysis_single_figure(axs[0, 1], all_results, 1)
    get_hyper_param_analysis_single_figure(axs[1, 0], all_results, 2)
    get_hyper_param_analysis_single_figure(axs[1, 1], all_results, 0)

    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    fig.text(0.04, 0.5, 'Results of Metrics', va="center", rotation="vertical")
    # fig.suptitle(f'{dataset_name}')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', frameon=False, ncol=4)

    plt.savefig(f'Hyperparameter Analysis on {dataset_name}.png')
    plt.show()



if __name__ == '__main__':
    # for cur_hyper_param_results in all_results:
    #     dataset_name = cur_hyper_param_results['dataset_name']
    #     hyper_param_name = cur_hyper_param_results['hyper_param_name']
    #     hyper_param_values = cur_hyper_param_results['hyper_param_values']
    #     results = cur_hyper_param_results['results']
    #     print_hyper_param_analysis_single_figure(dataset_name, hyper_param_name, hyper_param_values, results)
    print_hyper_param_analysis_multi_figure('Gowalla')
