class Metric:
    FID = 'fid'
    IS = 'is'
    FAKE_IM_MEAN = 'fake_im_mean'
    IS_MIXED = 'is_mixed'
    RANDOM = 'random'
    D_LOSS = 'd_loss'
    IF = 'isolation_forest'
    PRECISION = 'precision_k_5'
    RECALL = 'recall_k_5'
    DENSITY = 'density_k_5'
    COVERAGE = 'coverage_k_5'


D_DATASET_PATH = {
    'afhq_v2_cat': './data/afhq/cat/',
    'afhq_v2_dog': './data/afhq/dog/',
    'afhq_v2_wild': './data/afhq/wild/',
}

D_METRIC_LABEL = {
    Metric.FID: 'FID',
    Metric.IS: 'Inception Score',
    Metric.FAKE_IM_MEAN: 'Mean of Fake Images',
    Metric.IS_MIXED: 'Contaminated Labels',
    Metric.PRECISION: 'Precision',
    Metric.RECALL: 'Recall',
    Metric.DENSITY: 'Density',
    Metric.COVERAGE: 'Coverage',
}

class SetMetric:
    NEGATIVE = {Metric.FID, Metric.FAKE_IM_MEAN, Metric.RANDOM, Metric.D_LOSS, Metric.IF}
    POSITIVE = {Metric.IS_MIXED, Metric.IS}