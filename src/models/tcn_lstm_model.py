# -*- coding: utf-8 -*-
"""IET 2026 기반 TCN → LSTM Hybrid (구현가이드 23장).

TCN 블록이 시점별 Temporal Feature 시퀀스를 만들고, LSTM 이 그 시퀀스를 받아 장기 관계를 학습한다.
"""

from tensorflow import keras
from tensorflow.keras import layers

from src.models.tcn_model import resolve_dilations, tcn_stack


def build_tcn_lstm(
    input_shape,
    filters=64,
    kernel_size=3,
    dilations="auto",
    lstm_units=64,
    dropout=0.2,
    learning_rate=1e-3,
):
    lookback = input_shape[0]
    dilations = resolve_dilations(dilations, kernel_size, lookback)

    inputs = keras.Input(shape=input_shape)
    x = tcn_stack(inputs, filters, kernel_size, dilations, dropout)
    x = layers.LSTM(lstm_units)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name="tcn_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
