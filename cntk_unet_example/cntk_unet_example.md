

```python
import numpy as np
import cntk_unet
import simulation
%matplotlib inline
import helper

import cntk as C
from cntk.learners import learning_rate_schedule, UnitType
```


```python
# Generate some random images
input_images, target_masks = simulation.generate_random_data(shape=(1, 128, 128), count=3)

# Left: Input image, Right: Target mask
helper.plot_side_by_side([input_images, target_masks])
```


![png](output_1_0.png)



```python
from cntk.device import set_default_device, gpu
set_default_device(gpu(0))

def slice_minibatch(data_x, data_y, i, minibatch_size):
    sx = data_x[i * minibatch_size:(i + 1) * minibatch_size]
    sy = data_y[i * minibatch_size:(i + 1) * minibatch_size]
    
    return sx, sy

def measure_error(data_x, data_y, x, y, trainer, minibatch_size):
    errors = []
    for i in range(0, int(len(data_x) / minibatch_size)):
        data_sx, data_sy = slice_minibatch(data_x, data_y, i, minibatch_size)        

        errors.append(trainer.test_minibatch({x: data_sx, y: data_sy}))

    return np.mean(errors)

def train(images, masks, use_existing=False):  
    shape = input_images[0].shape
    data_size = input_images.shape[0]
    
    # Split data
    test_portion = int(data_size * 0.1)
    indices = np.random.permutation(data_size)
    test_indices = indices[:test_portion]
    training_indices = indices[test_portion:]

    test_data = (images[test_indices], masks[test_indices])
    training_data = (images[training_indices], masks[training_indices])

    # Create model
    x = C.input_variable(shape)
    y = C.input_variable(shape)

    z = cntk_unet.create_model(x)
    dice_coef = cntk_unet.dice_coefficient(z, y)

    # Load the saved model if specified
    checkpoint_file = "cntk-unet.dnn"
    if use_existing:
        z.load_model(checkpoint_file)

    # Prepare model and trainer
    lr = learning_rate_schedule(0.00001, UnitType.sample)
    momentum = C.learners.momentum_as_time_constant_schedule(0)
    trainer = C.Trainer(z, (-dice_coef, -dice_coef), C.learners.adam(z.parameters, lr=lr, momentum=momentum))

    # Get minibatches of training data and perform model training
    minibatch_size = 8
    num_epochs = 40
    
    training_errors = []
    test_errors = []

    for e in range(0, num_epochs):
        for i in range(0, int(len(training_data[0]) / minibatch_size)):
            data_x, data_y = slice_minibatch(training_data[0], training_data[1], i, minibatch_size)

            trainer.train_minibatch({x: data_x, y: data_y})
        
        # Measure training error
        training_error = measure_error(training_data[0], training_data[1], x, y, trainer, minibatch_size)
        training_errors.append(training_error)
        
        # Measure test error
        test_error = measure_error(test_data[0], test_data[1], x, y, trainer, minibatch_size)
        test_errors.append(test_error)

        print("epoch #{}: training_error={}, test_error={}".format(e, training_errors[-1], test_errors[-1]))
        
        trainer.save_checkpoint(checkpoint_file)

    return trainer, training_errors, test_errors
```


```python
input_images, target_masks = simulation.generate_random_data(shape=(1, 128, 128), count=512)

trainer, training_errors, test_errors = train(input_images, target_masks)
```

    epoch #0: training_error=-0.015713154959181946, test_error=-0.014074754590789476
    epoch #1: training_error=-0.016083419927510254, test_error=-0.014403479328999916
    epoch #2: training_error=-0.016407359081010025, test_error=-0.014691384974867105
    epoch #3: training_error=-0.01674052703668151, test_error=-0.014987555332481861
    epoch #4: training_error=-0.017066323002310174, test_error=-0.015277911753704151
    epoch #5: training_error=-0.017429420218002378, test_error=-0.015601862066735825
    epoch #6: training_error=-0.017894921840675045, test_error=-0.016016616486012936
    epoch #7: training_error=-0.018523585655840866, test_error=-0.016575909064461786
    epoch #8: training_error=-0.01939106546342373, test_error=-0.017347229178994894
    epoch #9: training_error=-0.020632880674511717, test_error=-0.018451130017638206
    epoch #10: training_error=-0.022468087737236107, test_error=-0.020082140651841957
    epoch #11: training_error=-0.025165321043970294, test_error=-0.022482942479352157
    epoch #12: training_error=-0.02855601730315309, test_error=-0.025510212096075218
    epoch #13: training_error=-0.032653404203684705, test_error=-0.029184202042718727
    epoch #14: training_error=-0.048196719795988316, test_error=-0.04313345874349276
    epoch #15: training_error=-0.5939852135223255, test_error=-0.5419719119866689
    epoch #16: training_error=-0.6386726493375343, test_error=-0.581634134054184
    epoch #17: training_error=-0.6611940405870739, test_error=-0.604099045197169
    epoch #18: training_error=-0.6818929408725939, test_error=-0.6256542503833771
    epoch #19: training_error=-0.7000348505220915, test_error=-0.6460102299849192
    epoch #20: training_error=-0.7140306684008816, test_error=-0.6648939947287241
    epoch #21: training_error=-0.7277951115056088, test_error=-0.6783098379770914
    epoch #22: training_error=-0.7416938574690568, test_error=-0.6878345608711243
    epoch #23: training_error=-0.7532186267668741, test_error=-0.6899469097455343
    epoch #24: training_error=-0.7626167347556666, test_error=-0.692071775595347
    epoch #25: training_error=-0.7662118485099391, test_error=-0.6867532432079315
    epoch #26: training_error=-0.77967888012267, test_error=-0.6946523686250051
    epoch #27: training_error=-0.7916771403530187, test_error=-0.6935372451941172
    epoch #28: training_error=-0.8172350473571242, test_error=-0.7144249379634857
    epoch #29: training_error=-0.7527119423213758, test_error=-0.6904274920622507
    epoch #30: training_error=-0.7842961256964165, test_error=-0.6523609856764475
    epoch #31: training_error=-0.8616938538718641, test_error=-0.7380031148592631
    epoch #32: training_error=-0.8811344525270295, test_error=-0.7324000497659048
    epoch #33: training_error=-0.8981306793396933, test_error=-0.7472967902819315
    epoch #34: training_error=-0.9092422765597963, test_error=-0.7710518936316172
    epoch #35: training_error=-0.9080749425971717, test_error=-0.7973848481973013
    epoch #36: training_error=-0.9296601508793078, test_error=-0.828477015097936
    epoch #37: training_error=-0.9297440658535874, test_error=-0.8322751522064209
    epoch #38: training_error=-0.9339224685702407, test_error=-0.8402199745178223
    epoch #39: training_error=-0.9440239260071203, test_error=-0.8584941625595093



```python
helper.plot_errors({"training": training_errors, "test": test_errors}, title="Simulation Learning Curve")
```


![png](output_4_0.png)



```python
# Generate some random images
input_images, target_masks = simulation.generate_random_data(shape=(1, 128, 128), count=5)

# Predict
pred = trainer.model.eval(input_images)
```


```python
# Left: Input image, Middle: Correct mask (Ground-truth), Rigth: Predicted mask
helper.plot_side_by_side([input_images, target_masks, pred[0]])
```


![png](output_6_0.png)

