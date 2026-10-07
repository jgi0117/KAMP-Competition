# -*- coding: utf-8 -*-
"""Conv1D 기반 TCN (구현가이드 15~16장).

출력은 마지막 시점(t-1)의 특징을 사용한다. causal padding 이므로 마지막 시점의 특징이
receptive field 안의 과거 전체를 요약한다.
"""

from tensorflow import keras
from tensorflow.keras import layers


def receptive_field(kernel_size, dilations):
    """residual block 당 causal Conv1D 2개 기준 receptive field (시간 수)."""
    return 1 + 2 * (kernel_size - 1) * sum(dilations)


def resolve_dilations(dilations, kernel_size, lookback):
    """'auto' 는 receptive field 가 lookback 이상이 되는 가장 짧은 [1, 2, 4, ...],
    'auto+1' 은 거기에 한 단계를 더한 것. 리스트는 그대로 사용한다."""
    if isinstance(dilations, str):
        extra = 1 if dilations == "auto+1" else 0
        result = [1]
        while receptive_field(kernel_size, result) < lookback:
            result.append(result[-1] * 2)
        for _ in range(extra):
            result.append(result[-1] * 2)
        return result
    return list(dilations)


def residual_tcn_block(x, filters, kernel_size, dilation_rate, dropout):
    shortcut = x

    x = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                      padding="causal", activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                      padding="causal")(x)

    # Residual shape 맞추기
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)

    x = layers.Add()([x, shortcut])
    return layers.Activation("relu")(x)


def tcn_stack(x, filters, kernel_size, dilations, dropout):
    for d in dilations:
        x = residual_tcn_block(x, filters, kernel_size, d, dropout)
    return x


def last_step(x, lookback):
    """(batch, lookback, ch) → (batch, ch). 직렬화 가능한 레이어만 사용."""
    x = layers.Cropping1D(cropping=(lookback - 1, 0))(x)
    return layers.Flatten()(x)


def build_tcn(
    input_shape,
    filters=64,
    kernel_size=3,
    dilations="auto",
    dropout=0.2,
    learning_rate=1e-3,
):
    lookback = input_shape[0]
    dilations = resolve_dilations(dilations, kernel_size, lookback)

    inputs = keras.Input(shape=input_shape)
    x = tcn_stack(inputs, filters, kernel_size, dilations, dropout)
    x = last_step(x, lookback)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name="tcn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
