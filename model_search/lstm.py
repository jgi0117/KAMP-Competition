SPACE = {
    "base": dict(lookback=24, hidden_units=64, num_layers=1, dropout=0.2,
                 learning_rate=1e-3, batch_size=32),
    "stages": [
        {"lookback": [24, 48, 72, 168]},
        {"hidden_units": [32, 64, 128], "num_layers": [1, 2]},
        {"learning_rate": [1e-4, 3e-4, 1e-3], "dropout": [0.0, 0.1, 0.2, 0.3]},
        {"batch_size": [16, 32, 64]},
    ],
}
