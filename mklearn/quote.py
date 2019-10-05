def random_quote():
    k = 1
    db = {1: '''Statistics tells us that if we need N samples for one feature, we need N^10 for 10 features. But if we assume independence amoung the features, then our N^10 data points get reduced to 10*N - Machine Learning in Action'''}
    return db[k]