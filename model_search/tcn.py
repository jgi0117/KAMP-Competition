SPACE = {
    "base": dict(lookback=24, filters=64, kernel_size=3, dilations="auto",
                 dropout=0.2, learning_rate=1e-3, batch_size=32),
    "stages": [
        {"lookback": [24, 48, 72, 168]},
        {"filters": [32, 64, 128], "kernel_size": [2, 3, 5], "dilations": ["auto", "auto+1"]},
        {"learning_rate": [1e-4, 3e-4, 1e-3], "dropout": [0.0, 0.1, 0.2, 0.3]},
        {"batch_size": [16, 32, 64]},
    ],
}
