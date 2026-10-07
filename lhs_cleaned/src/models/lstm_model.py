# -*- coding: utf-8 -*-
"""LSTM 기본 모델 (구현가이드 13장)."""

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm(
    input_shape,
    hidden_units=64,
    num_layers=1,
    dropout=0.2,
    learning_rate=1e-3,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for i in range(num_layers):
        x = layers.LSTM(hidden_units, return_sequences=i < num_layers - 1)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name="lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
