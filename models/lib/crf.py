'''This is a copy from the tf.contrib.crf package use only
    for inferior tensorflow versions

'''

import tensorflow as tf

def crf_decode(potentials, transition_params, sequence_length):
    """Decode the highest scoring sequence of tags in TensorFlow.
    This is a function for tensor. 

    -- Copied from tf.1.11 use this only if it's not provided by lib
    Args:
    potentials: A [batch_size, max_seq_len, num_tags] tensor of
              unary potentials.
    transition_params: A [num_tags, num_tags] matrix of
              binary potentials.
    sequence_length: A [batch_size] vector of true sequence lengths.
    Returns:
    decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
                Contains the highest scoring tag indices.
    best_score: A [batch_size] vector, containing the score of `decode_tags`.
    """

    # If max_seq_len is 1, we skip the algorithm and simply
    # return the argmax tag  and the max activation.
    def _single_seq_fn():
        squeezed_potentials = tf.squeeze(potentials, [1])
        decode_tags = tf.expand_dims(tf.argmax(squeezed_potentials, axis=1), 1)
        best_score = tf.reduce_max(squeezed_potentials, axis=1)

        return tf.cast(decode_tags, dtype=tf.int32), best_score

    def _multi_seq_fn():
        """Decoding of highest scoring sequence."""

        # For simplicity, in shape comments, denote:
        # 'batch_size' by 'B',
        # 'max_seq_len' by 'T' ,
        # 'num_tags' by 'O' (output).

        num_tags = potentials.get_shape()[2].value

        # Computes forward decoding. Get last score and backpointers.
        crf_fwd_cell = _CrfDecodeForwardRnnCell(transition_params)
        initial_state = tf.slice(potentials, [0, 0, 0], [-1, 1, -1])
        initial_state = tf.squeeze(initial_state, axis=[1])  # [B, O]
        inputs = tf.slice(potentials, [0, 1, 0], [-1, -1, -1])  # [B, T-1, O]
        # Sequence length is not allowed to be less than zero.
        sequence_length_less_one = tf.maximum(
            tf.constant(0, dtype=sequence_length.dtype),
            sequence_length - 1)
        backpointers, last_score = tf.nn.dynamic_rnn(  # [B, T - 1, O], [B, O]
            crf_fwd_cell,
            inputs=inputs,
            sequence_length=sequence_length_less_one,
            initial_state=initial_state,
            time_major=False,
            dtype=tf.int32)
        backpointers = tf.reverse_sequence(  # [B, T - 1, O]
            backpointers, sequence_length_less_one, seq_dim=1)

        # Computes backward decoding. Extract tag indices from backpointers.
        crf_bwd_cell = _CrfDecodeBackwardRnnCell(num_tags)
        initial_state = tf.cast(tf.argmax(last_score, axis=1),  # [B]
                                dtype=tf.int32)
        initial_state = tf.expand_dims(initial_state, axis=-1)  # [B, 1]
        decode_tags, _ = tf.nn.dynamic_rnn(  # [B, T - 1, 1]
            crf_bwd_cell,
            inputs=backpointers,
            sequence_length=sequence_length_less_one,
            initial_state=initial_state,
            time_major=False,
            dtype=tf.int32)
        decode_tags = tf.squeeze(decode_tags, axis=[2])  # [B, T - 1]
        decode_tags = tf.concat([initial_state, decode_tags],   # [B, T]
                                axis=1)
        decode_tags = tf.reverse_sequence(  # [B, T]
            decode_tags, sequence_length, seq_dim=1)

        best_score = tf.reduce_max(last_score, axis=1)  # [B]

        return decode_tags, best_score

    return tf.cond(
        pred=tf.equal(potentials.shape[1].value or tf.shape(potentials)[1], 1),
        true_fn=_single_seq_fn,
        false_fn=_multi_seq_fn)


class _CrfDecodeBackwardRnnCell(tf.nn.rnn_cell.RNNCell):
    """Computes backward decoding in a linear-chain CRF.
    """

    def __init__(self, num_tags):
        """Initialize the CrfDecodeBackwardRnnCell.
        Args:
        num_tags: An integer. The number of tags.
        """
        self._num_tags = num_tags

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 1

    def __call__(self, inputs, state, scope=None):
        """Build the CrfDecodeBackwardRnnCell.
        Args:
        inputs: A [batch_size, num_tags] matrix of
            backpointer of next step (in time order).
        state: A [batch_size, 1] matrix of tag index of next step.
        scope: Unused variable scope of this cell.
        Returns:
        new_tags, new_tags: A pair of [batch_size, num_tags]
        tensors containing the new tag indices.
        """
        state = tf.squeeze(state, axis=[1])                # [B]
        batch_size = tf.shape(inputs)[0]
        b_indices = tf.range(batch_size)                    # [B]
        indices = tf.stack([b_indices, state], axis=1)     # [B, 2]
        new_tags = tf.expand_dims(
        tf.gather_nd(inputs, indices),             # [B]
        axis=-1)                                              # [B, 1]

        return new_tags, new_tags


class _CrfDecodeForwardRnnCell(tf.nn.rnn_cell.RNNCell):
    """Computes the forward decoding in a linear-chain CRF.
    """

    def __init__(self, transition_params):
        """Initialize the CrfDecodeForwardRnnCell.
        Args:
          transition_params: A [num_tags, num_tags] matrix of binary
            potentials. This matrix is expanded into a
            [1, num_tags, num_tags] in preparation for the broadcast
            summation occurring within the cell.
        """
        self._transition_params = tf.expand_dims(transition_params, 0)
        self._num_tags = transition_params.get_shape()[0].value

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None):
        """Build the CrfDecodeForwardRnnCell.
        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous step's
                score values.
          scope: Unused variable scope of this cell.
        Returns:
          backpointers: A [batch_size, num_tags] matrix of backpointers.
          new_state: A [batch_size, num_tags] matrix of new score values.
        """
        # For simplicity, in shape comments, denote:
        # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
        state = tf.expand_dims(state, 2)                         # [B, O, 1]

        # This addition op broadcasts self._transitions_params along the zeroth
        # dimension and state along the second dimension.
        # [B, O, 1] + [1, O, O] -> [B, O, O]
        transition_scores = state + self._transition_params             # [B, O, O]
        new_state = inputs + tf.reduce_max(transition_scores, [1])  # [B, O]
        backpointers = tf.argmax(transition_scores, 1)
        backpointers = tf.cast(backpointers, dtype=tf.int32)    # [B, O]

        return backpointers, new_state