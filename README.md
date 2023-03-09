# Shakespeare reborn

## Context

The goal of the project is to generate text based on a decoder that reproduces the style of Shakespeare

I based myself on the document https://arxiv.org/pdf/1706.03762v5.pdf to realize this AI.

The code is commented in the script_cpu.py.
The second script (script_gpu.py) has the same logic but is set up to run on GPUs.

##  /!\Before Running/!\

You need to make sure you have installed:

pip install transformers torch numpy datasets tiktoken wandb tqdm cuda-python numba

Depuis colab :
    - put the 3 files in the same path
    - Run script_cpu.py :

`!python script_cpu.py`


   - Run script_gpu.py :

Check the number of gpus available:

`
import torch
torch.__version__ 
torch.cuda.is_available() 
torch.cuda.device_count() 
torch.cuda.get_device_name(0)
`
- [x] Add more lines with +1 like get_device_name(1), get_device_name(2) if you have more devices.

And run the command :
`!torchrun --standalone --nproc_per_node=<nb_gpu> script_gpu.py`

## Optimizations - Next steps

- [ ] Tunning hyperparameters with cuda
- [ ] Create machine translation : encoder with cross attention block
- [ ] Create gpt2 : saving/loading checkpoint, pre-trained weigths, implement a batch manner (4 dimensions array tensors), implement multiply perceptron

## OUTPUT

### script_cpu.py

With 32 Go RAM:
 running < 4 min
step 0: train losse 4.6340, val loss : 4.6524
step 100: train losse 2.5227, val loss : 2.5400
step 200: train losse 2.3092, val loss : 2.3167
step 300: train losse 2.1279, val loss : 2.1822
step 400: train losse 2.0245, val loss : 2.1227
step 500: train losse 1.9807, val loss : 2.0675
step 600: train losse 1.9053, val loss : 2.0332
step 700: train losse 1.8642, val loss : 1.9924
step 800: train losse 1.8307, val loss : 1.9694
step 900: train losse 1.8106, val loss : 1.9309
step 1000: train losse 1.8037, val loss : 1.9239
step 1100: train losse 1.7449, val loss : 1.9011
step 1200: train losse 1.7252, val loss : 1.8898
step 1300: train losse 1.7060, val loss : 1.8799
step 1400: train losse 1.6998, val loss : 1.8670
step 1500: train losse 1.7028, val loss : 1.8418
step 1600: train losse 1.6455, val loss : 1.8723
step 1700: train losse 1.6291, val loss : 1.8199
step 1800: train losse 1.6357, val loss : 1.8061
step 1900: train losse 1.6366, val loss : 1.8152

LEONES:
You aet shall, sille at with a well:
Out duke?' love! shull what that unk'd santeh?' To
That thy trunk that you,
To like holy, I very thou will stin's tose that the crother
tess you?
I have hope sleasided any of malb: yet stray,
Doset you, are shephis much a dispers,
I word, by the noble a bose not for not vattep's thea,
To dost thou not in the duke till't have them ever.

JERCANTIO:
A soset live, you, a follows two show prive,
My for to-us to not you doot warwinks
Nor with that five blood speer enoused of love
It beald by: what they balieven'd such aloved to beauty:
It let who sep and strike how this in the fulat he duket
To me.

KING RICHARD III:
My larvinghisfficlament must you have,
Eve laive thou nort reteels at must love thusfort
That lave perliof still many faice:
Ay, but parither'd he world, pay 
Master my lord's viemthal man?
Head prethe not that you not Ge!

### script_gpu.py

With 1 GPU :
 running < 32 min

step 0: train losse 4.6626, val loss : 4.6560
step 100: train losse 2.5551, val loss : 2.5769
step 200: train losse 2.4813, val loss : 2.5017
step 300: train losse 2.4356, val loss : 2.4614
step 400: train losse 2.3511, val loss : 2.3779
step 500: train losse 2.1917, val loss : 2.2354
step 600: train losse 2.0804, val loss : 2.1465
step 700: train losse 1.9750, val loss : 2.0652
step 800: train losse 1.9037, val loss : 2.0097
step 900: train losse 1.8377, val loss : 1.9656
step 1000: train losse 1.7631, val loss : 1.9143
step 1100: train losse 1.7283, val loss : 1.8675
step 1200: train losse 1.6912, val loss : 1.8365
step 1300: train losse 1.6540, val loss : 1.8133
step 1400: train losse 1.6302, val loss : 1.8121
step 1500: train losse 1.6105, val loss : 1.7789
step 1600: train losse 1.5844, val loss : 1.7730
step 1700: train losse 1.5575, val loss : 1.7481
step 1800: train losse 1.5430, val loss : 1.7357
step 1900: train losse 1.5318, val loss : 1.7385

VOLYCA:
Mamen, terry, jordices spirit to thee thee, they
Of to gue there is they houst.

Ficest Lord Of Gove a good's coot.
The man of lady? awher let,
God reven I 
Are you a very atmany. Ho, lord, fiery
tello on that proful forted as pridined
Of this sefit exper; and yet chongers.

CORIOLANUS:
Sir, and he his noble.

KINCHARI:
Master, if my love.
