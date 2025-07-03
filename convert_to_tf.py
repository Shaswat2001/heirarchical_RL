import tensorflow as tf
from tensorflow.keras import layers
print(tf.config.list_physical_devices('GPU'))
import tensorflow as tf

import tensorflow as tf

class MLP(tf.keras.Model):
    def __init__(self, hidden_layers, activation=tf.nn.relu, activate_final=True, layer_norm=True, kernel_init='glorot_uniform'):
        super(MLP, self).__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.activate_final = activate_final
        self.layer_norm = layer_norm
        
        # Initialize layers
        self.dense_layers = []
        for size in self.hidden_layers:
            self.dense_layers.append(tf.keras.layers.Dense(size, kernel_initializer=kernel_init))
            if self.layer_norm:
                self.dense_layers.append(tf.keras.layers.LayerNormalization())
        
        # The final output layer (if required)
        self.final_activation = None
        if self.activate_final:
            self.final_activation = self.activation
        
    def call(self, inputs, training=False):
        x = inputs
        for i, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x)
            if i < len(self.dense_layers) - 1:  # Apply activation and layer norm except for the last one
                x = self.activation(x)
                if self.layer_norm and isinstance(dense_layer, tf.keras.layers.LayerNormalization):
                    x = dense_layer(x, training=training)

        # Apply final activation if specified
        if self.final_activation:
            x = self.final_activation(x)
        
        return x

class GCDetActor(tf.keras.Model):
    def __init__(self, hidden_layers, action_dim, final_fc_init_scale=1e-2):
        super(GCDetActor, self).__init__()
        self.actor_net = MLP(hidden_layers, activate_final=True, layer_norm=True)
        self.action_net = tf.keras.layers.Dense(action_dim, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=final_fc_init_scale))
    
    def call(self, observations, goal=None, temperature=1.0):
        # Concatenate observations and goal if goal is provided
        inputs = [observations]
        if goal is not None:
            inputs.append(goal)
        
        # Concatenate along the last axis (equivalent to Flax's jnp.concatenate)
        inputs = tf.concat(inputs, axis=-1)
        
        # Pass through the actor network (MLP)
        outputs = self.actor_net(inputs)
        
        # Pass through the action network and apply tanh
        means = self.action_net(outputs)
        return tf.tanh(means)
    
import keras
import numpy as np

def convert_flax_actor_to_keras(actor_param, input_dim, hidden_sizes, action_dim, final_fc_init_scale=1e-2):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(shape=(input_dim,)))

    # Add actor_net layers
    for i, size in enumerate(hidden_sizes):
        model.add(keras.layers.Dense(size, activation='relu', name=f"Dense_{i}"))
        model.add(keras.layers.LayerNormalization(name=f"LayerNorm_{i}"))

    # Add final action_net Dense + tanh activation
    model.add(keras.layers.Dense(
        action_dim,
        activation='tanh',
        kernel_initializer=keras.initializers.RandomNormal(stddev=final_fc_init_scale),
        name="action_net"
    ))

    # === Assign weights ===
    # actor_net weights
    actor_net = actor_param['actor_net']
    flax_layer_names = ['Dense_0', 'LayerNorm_0', 'Dense_1', 'LayerNorm_1']
    keras_layers = model.layers[0:-1]  # skip input, exclude final Dense for now

    for name, layer in zip(flax_layer_names, keras_layers):
        flax_weights = actor_net[name]
        print(name)
        print(layer)
        if isinstance(layer, keras.layers.Dense):
            w = np.array(flax_weights['kernel'])
            b = np.array(flax_weights['bias'])
            layer.set_weights([w, b])

        elif isinstance(layer, keras.layers.LayerNormalization):
            print(flax_weights)
            scale = np.array(flax_weights['scale'])
            bias = np.array(flax_weights['bias'])
            # For inference, moving stats can be dummy
            layer.set_weights([scale, bias])

    # action_net weights
    action_net = actor_param['action_net']
    final_layer = model.layers[-1]
    w = np.array(action_net['kernel'])
    b = np.array(action_net['bias'])
    final_layer.set_weights([w, b])

    return model

    
import pickle

with open("/home/ubuntu/uploads/heirarchical_RL/exp/hrl-arenaX/Debug/FrankaIkGolfCourseEnv_20250612-235606_gcbc/params_400000.pkl", "rb") as f:
    flax_params = pickle.load(f)

# actor_param = flax_params['agent']["network"]["params"]["modules_actor"]["actor_net"]

actor_param = flax_params['agent']["network"]["params"]["modules_actor"]
print(actor_param["action_net"].keys())

model = convert_flax_actor_to_keras(
    actor_param=actor_param,
    input_dim=62,              # 31 obs + 31 goal
    hidden_sizes=[512, 512],
    action_dim=7
)

model.save("converted_tf1_model.keras")

import numpy as np

# tf_model = GCDetActor(hidden_layers=(512, 512), action_dim=7)
# print(tf_model.layers)
# _ = tf_model(tf.random.normal([1, 31]), tf.random.normal([1, 31]))  # Build it

# # Assign weights from flax_params to tf_model layer by layer
# import numpy as np

# # Flatten the Flax layers manually in the expected order
# flax_layer_names = ['Dense_0', 'LayerNorm_0', 'Dense_1', 'LayerNorm_1', 'Dense_2']  # Adjust as needed

# # Collect TensorFlow sublayers (Dense and LayerNorm)
# tf_layers = [layer for layer in tf_model.actor_net.layers if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.LayerNormalization))]

# for name, tf_layer in zip(flax_layer_names, tf_layers):
#     flax_layer = actor_param[name]

#     if isinstance(tf_layer, tf.keras.layers.Dense):
#         w = np.array(flax_layer['kernel'])  # shape: [in_dim, out_dim]
#         b = np.array(flax_layer['bias'])    # shape: [out_dim]
#         tf_layer.set_weights([w, b])

#     elif isinstance(tf_layer, tf.keras.layers.LayerNormalization):
#         scale = np.array(flax_layer['scale'])  # gamma
#         bias  = np.array(flax_layer['bias'])   # beta
#         tf_layer.set_weights([scale, bias])


# tf_model.save("converted_tf_model.keras")