"""UNETR implementation in Tensorflow 2.0.

Adapted from https://www.kaggle.com/code/usharengaraju/tensorflow-unetr-w-b
"""

import math

import tensorflow as tf


class SingleDeconv3DBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(SingleDeconv3DBlock, self).__init__()
        self.block = tf.keras.layers.Conv3DTranspose(
            filters=filters,
            kernel_size=2,
            strides=2,
            padding="valid",
            output_padding=None,
        )

    def call(self, inputs):
        return self.block(inputs)


class SingleConv3DBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(SingleConv3DBlock, self).__init__()
        self.kernel = kernel_size
        self.res = tuple(map(lambda i: (i - 1) // 2, self.kernel))
        self.block = tf.keras.layers.Conv3D(
            filters=filters, kernel_size=kernel_size, strides=1, padding="same"
        )

    def call(self, inputs):
        return self.block(inputs)


class Conv3DBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3, 3)):
        super(Conv3DBlock, self).__init__()
        self.a = tf.keras.Sequential(
            [
                SingleConv3DBlock(filters, kernel_size=kernel_size),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
            ]
        )

    def call(self, inputs):
        return self.a(inputs)


class Deconv3DBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3, 3)):
        super(Deconv3DBlock, self).__init__()
        self.a = tf.keras.Sequential(
            [
                SingleDeconv3DBlock(filters=filters),
                SingleConv3DBlock(filters=filters, kernel_size=kernel_size),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
            ]
        )

    def call(self, inputs):
        return self.a(inputs)


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, embed_dim, dropout):
        super(SelfAttention, self).__init__()

        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = tf.keras.layers.Dense(self.all_head_size)
        self.key = tf.keras.layers.Dense(self.all_head_size)
        self.value = tf.keras.layers.Dense(self.all_head_size)

        self.out = tf.keras.layers.Dense(embed_dim)
        self.attn_dropout = tf.keras.layers.Dropout(dropout)
        self.proj_dropout = tf.keras.layers.Dropout(dropout)

        self.softmax = tf.keras.layers.Softmax()

        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = list(
            x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        )
        new_x_shape[0] = -1
        y = tf.reshape(x, new_x_shape)
        return tf.transpose(y, perm=[0, 2, 1, 3])

    def call(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = query_layer @ tf.transpose(key_layer, perm=[0, 1, 3, 2])
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = attention_probs @ value_layer
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        new_context_layer_shape = list(context_layer.shape[:-2] + (self.all_head_size,))
        new_context_layer_shape[0] = -1
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(tf.keras.layers.Layer):
    def __init__(self, output_features, drop=0.0):
        super(Mlp, self).__init__()
        self.a = tf.keras.layers.Dense(units=output_features, activation=tf.nn.gelu)
        self.b = tf.keras.layers.Dropout(drop)

    def call(self, inputs):
        x = self.a(inputs)
        return self.b(x)


class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model=768, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.a = tf.keras.layers.Dense(units=d_ff)
        self.b = tf.keras.layers.Dense(units=d_model)
        self.c = tf.keras.layers.Dropout(dropout)

    def call(self, inputs):
        return self.b(self.c(tf.nn.relu(self.a(inputs))))


# embeddings, projection_dim=embed_dim
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, cube_size, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.num_of_patches = int(
            (cube_size[0] * cube_size[1] * cube_size[2])
            / (patch_size * patch_size * patch_size)
        )
        self.patch_size = patch_size
        self.size = patch_size
        self.embed_dim = embed_dim

        self.projection = tf.keras.layers.Dense(embed_dim)

        self.clsToken = tf.Variable(
            tf.keras.initializers.GlorotNormal()(shape=(1, 512, embed_dim)),
            trainable=True,
        )

        self.positionalEmbedding = tf.keras.layers.Embedding(
            self.num_of_patches, embed_dim
        )
        self.patches = None
        self.lyer = tf.keras.layers.Conv3D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
        )
        # embedding - basically is adding numerical embedding to the layer along with an extra dim

    def call(self, inputs):
        patches = self.lyer(inputs)
        patches = tf.reshape(
            patches, (tf.shape(inputs)[0], -1, self.size * self.size * 3)
        )
        patches = self.projection(patches)
        positions = tf.range(0, self.num_of_patches, 1)[tf.newaxis, ...]
        positionalEmbedding = self.positionalEmbedding(positions)
        patches = patches + positionalEmbedding

        return patches, positionalEmbedding


# transformerblock
class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super(TransformerLayer, self).__init__()

        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.mlp_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # embed_dim/no-of_heads
        self.mlp_dim = int(
            (cube_size[0] * cube_size[1] * cube_size[2])
            / (patch_size * patch_size * patch_size)
        )

        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def call(self, x, training=True):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h

        return x, weights


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        cube_size,
        patch_size,
        num_layers=12,
        dropout=0.1,
        extract_layers=[3, 6, 9, 12],
    ):
        super(TransformerEncoder, self).__init__()
        #  embed_dim, num_heads ,dropout, cube_size, patch_size
        self.embeddings = PatchEmbedding(cube_size, patch_size, embed_dim)
        self.extract_layers = extract_layers
        self.encoders = [
            TransformerLayer(embed_dim, num_heads, dropout, cube_size, patch_size)
            for _ in range(num_layers)
        ]

    def call(self, inputs, training=True):
        extract_layers = []
        x = inputs
        x, _ = self.embeddings(x)

        for depth, layer in enumerate(self.encoders):
            x, _ = layer(x, training=training)
            if depth + 1 in self.extract_layers:
                extract_layers.append(x)

        return extract_layers


class UNETR(tf.keras.Model):
    def __init__(
        self,
        img_shape=(96, 96, 96),
        input_dim=3,
        output_dim=3,
        embed_dim=768,
        patch_size=16,
        num_heads=12,
        dropout=0.1,
    ):
        super(UNETR, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]
        self.transformer = TransformerEncoder(
            self.embed_dim,
            self.num_heads,
            self.img_shape,
            self.patch_size,
            self.num_layers,
            self.dropout,
            self.ext_layers,
        )

        # U-Net Decoder
        self.decoder0 = tf.keras.Sequential(
            [Conv3DBlock(32, (3, 3, 3)), Conv3DBlock(64, (3, 3, 3))]
        )

        self.decoder3 = tf.keras.Sequential(
            [Deconv3DBlock(512), Deconv3DBlock(256), Deconv3DBlock(128)]
        )

        self.decoder6 = tf.keras.Sequential([Deconv3DBlock(512), Deconv3DBlock(256)])

        self.decoder9 = Deconv3DBlock(512)

        self.decoder12_upsampler = SingleDeconv3DBlock(512)

        self.decoder9_upsampler = tf.keras.Sequential(
            [
                Conv3DBlock(512),
                Conv3DBlock(512),
                Conv3DBlock(512),
                SingleDeconv3DBlock(256),
            ]
        )

        self.decoder6_upsampler = tf.keras.Sequential(
            [Conv3DBlock(256), Conv3DBlock(256), SingleDeconv3DBlock(128)]
        )

        self.decoder3_upsampler = tf.keras.Sequential(
            [Conv3DBlock(128), Conv3DBlock(128), SingleDeconv3DBlock(64)]
        )

        self.decoder0_header = tf.keras.Sequential(
            [Conv3DBlock(64), Conv3DBlock(64), SingleConv3DBlock(output_dim, (1, 1, 1))]
        )

    def call(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, z[0], z[1], z[2], z[3]
        z3 = tf.reshape(
            tf.transpose(z3, perm=[0, 2, 1]), [-1, *self.patch_dim, self.embed_dim]
        )
        z6 = tf.reshape(
            tf.transpose(z6, perm=[0, 2, 1]), [-1, *self.patch_dim, self.embed_dim]
        )
        z9 = tf.reshape(
            tf.transpose(z9, perm=[0, 2, 1]), [-1, *self.patch_dim, self.embed_dim]
        )
        z12 = tf.reshape(
            tf.transpose(z12, perm=[0, 2, 1]), [-1, *self.patch_dim, self.embed_dim]
        )
        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(tf.concat([z9, z12], 4))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(tf.concat([z6, z9], 4))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(tf.concat([z3, z6], 4))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(tf.concat([z0, z3], 4))
        return output

    # def model(self):
    #     x = tf.keras.layers.Input(shape=(96, 96, 96, 3))
    #     return tf.keras.Model(inputs=[x], outputs=self.call(x))


def unetr(
    n_classes=1,
    input_shape=(96, 96, 96, 3),
    embed_dim=768,
    patch_size=16,
    num_heads=12,
    dropout=0.1,
):
    *img_shape, input_dim = input_shape

    input = tf.keras.layers.Input([*img_shape, input_dim], name="input_image")

    z = UNETR(
        img_shape=img_shape,
        input_dim=input_dim,
        output_dim=n_classes,
        embed_dim=embed_dim,
        patch_size=patch_size,
        num_heads=num_heads,
        dropout=dropout,
    )(input)

    final_activation = "sigmoid" if n_classes == 1 else "softmax"
    output = tf.keras.layers.Activation(final_activation)(z)

    return tf.keras.Model(inputs=[input], outputs=[output])


if __name__ == "__main__":
    input_shape = (96, 96, 96, 3)
    sub1 = unetr(input_shape=input_shape, n_classes=1)
    sub1.summary()
