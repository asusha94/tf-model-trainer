import tensorflow as tf


class ModelBuilder:
    def __init__(self):
        self._forward_fn = None
        self._loss_fn = None
        self._exclude_params = None

    def set_exclude_params(self, exclude_params):
        # callable or list or tuple or None
        self._exclude_params = exclude_params
        return self

    def set_forward(self, forward):
        if not callable(forward):
            raise ValueError('forward: is not callable')

        self._forward_fn = forward
        return self

    def set_loss(self, loss):
        if not callable(loss):
            raise ValueError('loss: is not callable')
            
        self._loss_fn = loss
        return self

    def build(self):
        forward_fn = self._forward_fn
        loss_fn = self._loss_fn
        exclude_params = self._exclude_params
        
        class AnonymousModel:
            def __init__(self):
                self._forward_fn = forward_fn
                self._loss_fn = loss_fn
                self._exclude_params = exclude_params

            def forward(self, is_training_mode, *inputs):
                self.inputs = inputs
                
                self.outputs = self._forward_fn(is_training_mode, *self.inputs)
                    
                params = tf.trainable_variables()

                if callable(self._exclude_params):
                    exclude_params = self._exclude_params()
                else:
                    exclude_params = self._exclude_params

                if exclude_params:
                    params = [param for param in params if param.name not in exclude_params]

                self.params = params

                return self.outputs

            def loss(self, scope=None):
                outputs = self.outputs
                if not isinstance(outputs, (tuple, list)):
                    outputs = [outputs]
                    
                self._loss_fn(*self.inputs, *outputs)
                
                losses = tf.losses.get_losses(scope=scope)
                reg_losses = tf.losses.get_regularization_losses(scope=scope)

                self._loss = tf.add_n(losses)
                if len(reg_losses):
                    self._loss = self._loss + tf.add_n(reg_losses)
                
                self.losses = losses
                self.reg_losses = reg_losses

                return losses

            def gradients(self):
                _params = self.params

                grads = tf.gradients(self._loss, _params)
                return list(zip(grads, _params))
        
        return AnonymousModel
