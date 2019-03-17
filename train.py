import tensorflow as tf


def training_loop(model, loss_fn, train_dataset, num_epochs, **kwargs):

    metrics = kwargs.get('metrics', None)
    valid_dataset = kwargs.get('valid_dataset', None)
    log_every_n_steps = kwargs.get('log_every_n_steps', 10)
    return_learning_curves = kwargs.get('return_learning_curves', False)

    learning_rate = kwargs.get('learning_rate', 0.0001)
    optimizer = kwargs.get('optimizer', tf.train.AdamOptimizer(learning_rate=learning_rate))
    
    for epoch in range(num_epochs):

        for step, (images, labels) in enumerate(train_dataset):
                
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_fn(labels, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
            if step % log_every_n_steps == 0:
                print(f"Epoch: {epoch} | Step: {step} | Loss: {loss}")

    
