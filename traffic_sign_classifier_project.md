traffic\_sign\_classifier\_project file:///home/lazic/udacity/CarND-Traffic-Sign-Class...![](traffic\_sign\_classifier\_project.001.png)

Self-Driving Car Engineer Nanodegree

Deep Learning

Project: Build a Traffic Sign Recognition Classifier

In this notebook is an implementation of Traffic Sign Classifier using LeNet neural network for the Udacity Self Drivig Car Nanodegree program.

You can find this project on  this github repo

Below I will adress each point in the  project rubric .

The project files are:

'traffic\_sign\_classifier\_project.ipynb' is a jupyter notebook containing the code

'traffic\_sign\_classifier\_project.html' is the HTML export of the code 'traffic\_sign\_classifier\_project.pdf' is the project writeup in pdf

**import** pickle

**import** matplotlib.pyplot **as** plt

**import** random

**import** numpy **as** np

**import** csv

**import** warnings

**from** sklearn.utils **import** shuffle

**import** cv2

- Suppressing TensorFlow FutureWarings

**with** warnings **.** catch\_warnings ():

warnings **.** filterwarnings ( "ignore" , category **=**FutureWarning )

**import** tensorflow.compat.v1 **as** tf

tf **.** disable\_v2\_behavior ()

WARNING:tensorflow:From /home/lazic/.local/lib/python3.8/site-packages/tensor flow/python/compat/v2\_compat.py:96: disable\_resource\_variables (from tensorfl ow.python.ops.variable\_scope) is deprecated and will be removed in a future v ersion.

Instructions for updating:

non-resource variables are not supported in the long term

image\_label\_file **=** 'signnames.csv'

**def** parse\_image\_labels ( input\_csc\_file ):

reader **=** csv **.** reader ( open ( input\_csc\_file , 'r' )) retVal **=** {}

**for** row **in** reader :

key , value **=** row

**if** key **==** 'ClassId' :

**continue**

retVal **.** update ({ int ( key ): value })

**return** retVal

- Parsing image label csv file

image\_labels **=** parse\_image\_labels ( image\_label\_file ) Data set

In the two cells bellow several rubric points are addressed:

'Dataset Summary' in which I display the basic propertives of the dataset like the number of images, number of classes and image shape

'Exploratory Visualization' in which I display selected images of the dataset

'Preprocessing' where I apply preprocessing techniques to the dataset. The techniques I have implemented are dataset normalization and grayscale. My initial idea was to train the network using RGB images, so it was required that I normalize the values of image pixels in order to improve the the perscision of the network. However, after initial few tries I have opted againts this approach and went with the training the network on a grayscale input data set. This approach proved to be more effective in terms of achieving desired network percision.

In the cell below I load the train, validation and tests set and apply the grayscaleing on each image for each set.

"""

`    `Functions for dataset exploration

"""

figsize\_default **=** plt **.** rcParams [ 'figure.figsize' ] **def** samples\_stat ( features , labels ):

h **=** [ 0 **for** i **in** range ( len ( labels ))]

samples **=** {}

**for** idx , l **in** enumerate ( labels ):

h[ l ] **+=** 1

**if** l **not in** samples :

samples [ l ] **=** features [ idx ]

**return** h, samples

**def** dataset\_exploration ( features , labels ):

plt **.** rcParams [ 'figure.figsize' ] **=** ( 20.0 , 20.0 ) histo , samples **=** samples\_stat ( features , labels ) total\_class **=** len ( set ( labels ))

ncols **=** 4

nrows **=** 11

\_, axes **=** plt **.** subplots ( nrows **=**nrows , ncols **=**ncols )

class\_idx **=** 0

**for** r **in** range ( nrows ):

**for** c **in** range ( ncols ):

a **=** axes [ r ][ c] a**.** axis ( 'off' )

**if** class\_idx **in** samples :

a**.** imshow ( samples [ class\_idx ])

**if** class\_idx **in** image\_labels :

a**.** set\_title ( "No. {} {} (# {} )" **.** format ( class\_idx , image\_labels [ class

class\_idx **+=** 1

plt **.** rcParams [ 'figure.figsize' ] **=** figsize\_default
PAGE3 of NUMPAGES17 12/3/20, 1:40 PM
traffic\_sign\_classifier\_project file:///home/lazic/udacity/CarND-Traffic-Sign-Class...![](traffic\_sign\_classifier\_project.002.png)

training\_file **=** './data\_set/train.p' validation\_file **=** './data\_set/valid.p' testing\_file **=** './data\_set/test.p'

- Loading the data set

**with** open ( training\_file , mode**=**'rb' ) **as** f :

train **=** pickle **.** load ( f )

**with** open ( validation\_file , mode**=**'rb' ) **as** f :

valid **=** pickle **.** load ( f )

**with** open ( testing\_file , mode**=**'rb' ) **as** f :

test **=** pickle **.** load ( f )

X\_train , y\_train **=** train [ 'features' ], train [ 'labels' ] X\_valid , y\_valid **=** valid [ 'features' ], valid [ 'labels' ] X\_test , y\_test **=** test [ 'features' ], test [ 'labels' ]

**assert** ( len ( X\_train ) **==** len ( y\_train )) **assert** ( len ( X\_valid ) **==** len ( y\_valid ))

print ()

print ( "Image Shape:  {} "**.** format ( X\_train [ 0] **.** shape ))

print ()

print ( "Training Set:    {} samples" **.** format ( len ( X\_train ))) print ( "Validation Set:  {} samples" **.** format ( len ( X\_valid ))) print ( "Test Set:        {} samples" **.** format ( len ( X\_test )))

**def** image\_normalize ( image ):

image **=** np**.** divide ( image , 255 )

**return** image

**def** dataset\_normalization ( X\_data ):

X\_normalized **=** X\_data **.** copy ()

num\_examples **=** len ( X\_data )

**for** i **in** range ( num\_examples ):

image **=** X\_normalized [ i ]

normalized\_image **=** image\_normalize ( image ) X\_normalized [ i ] **=** normalized\_image

**return** X\_normalized

**def** dataset\_grayscale ( X\_data ):

X\_grayscale **=** []

num\_examples **=** len ( X\_data )

**for** i **in** range ( num\_examples ):

image **=** X\_data [ i ]

gray **=** cv2 **.** cvtColor ( image , cv2 **.** COLOR\_RGB2GRA)Y

X\_grayscale **.** append ( gray **.** reshape ( 32, 32, 1)) **return** np**.** array ( X\_grayscale )

dataset\_exploration ( X\_train , y\_train )

print ( 'Grayscaling training set' ) X\_train **=** dataset\_grayscale ( X\_train ) X\_valid **=** dataset\_grayscale ( X\_valid ) X\_test **=** dataset\_grayscale ( X\_test )

**assert** ( len ( X\_train ) **==** len ( y\_train ))

print ( "Grayscaled Training Set:    {} samples" **.** format ( len ( X\_train )))

4 of 13 print ( "Grayscale Image Shape:  {} "**.** format ( X\_train [ 0] **.** shape )) 12/3/20, 1:40 PM

traffic\_sign\_classifier\_project file:///home/lazic/udacity/CarND-Traffic-Sign-Class...

Image Shape: (32, 32, 3)

Training Set:   34799 samples

Validation Set: 4410 samples

Test Set:       12630 samples

Grayscaling training set

Grayscaled Training Set:   34799 samples Grayscale Image Shape: (32, 32, 1)

![](traffic\_sign\_classifier\_project.003.png)![](traffic\_sign\_classifier\_project.004.png)

Neural Network

In the two cells bellow several rubric points are addressed:

![](traffic\_sign\_classifier\_project.005.png) 'Model Architecture' : For model architecture I have opted for standard LeNet neural network. The neural network takes an input shape of 32x32x1, which is a grayscaled image. The network contains two convolutional layers. The convolutional layers are a combination of convolution, relu activation function, max pool layer. After that we have three fully connected layers. After each respective layers I have implemented a dropout layer, for reducing network overfitting. The dropout layers have 2 different keep probabilities. One
PAGE7 of NUMPAGES17 12/3/20, 1:40 PM
traffic\_sign\_classifier\_project file:///home/lazic/udacity/CarND-Traffic-Sign-Class...![](traffic\_sign\_classifier\_project.006.png)

probability is used for the output of convolutional layer, and the other is used for the output of fully connected layers. The output of the network is an array of logits size 43. The network

is implemented in lenet.py file.

'Model Training' : The training of the network was done on the input sample of the grayscaled images. The images have been preprocessed in the cells above. The network

has been trained with a learing rate of 0.001 over 30 epoch with a bach size of 64. For optimizer I used Adam optimizer.

'Solution Approach' : After a serveral tried tunning the hyperparameters I found the best was to use a learning rate of 0.001 and the batch size of 64. For the number of epoch I found that 30 number of epoch gave the network enough tries to achieve desired validation accuracy without having to take too much time or run the risk of the network overfitting. After 30 epoch the validation accuracy that was achieved was 94.6%. After getting the desired validation accuracy the network was then introduced to the test set. On the first try running the network on the tests set the accuracy achieved was 93.2% which is a good inicator that

the network was trained correctly and that it didn't overfit. **from** lenet **import \***

EPOCHS **=** 30 BATCH\_SIZE **=** 64

x **=** tf **.** placeholder ( tf **.** float32 , ( **None**, 32, 32, 1)) y **=** tf **.** placeholder ( tf **.** int32 , **None**)

one\_hot\_y **=** tf **.** one\_hot ( y, 43)

logits **=** LeNet ( x)

- Training pipeline rate **=** 0.001

cross\_entropy **=** tf **.** nn**.** softmax\_cross\_entropy\_with\_logits ( labels **=**one\_hot\_y , logit loss\_operation **=** tf **.** reduce\_mean ( cross\_entropy )

optimizer **=** tf **.** train **.** AdamOptimizer ( learning\_rate **=**rate )

training\_operation **=** optimizer **.** minimize ( loss\_operation )

- Model evaluation

correct\_prediction **=** tf **.** equal ( tf **.** argmax ( logits , 1), tf **.** argmax ( one\_hot\_y , 1)) accuracy\_operation **=** tf **.** reduce\_mean ( tf **.** cast ( correct\_prediction , tf **.** float32 ))

WARNING:tensorflow:From /home/lazic/.local/lib/python3.8/site-packages/tensor

flow/python/util/dispatch.py:201: calling dropout (from tensorflow.python.op

s.nn\_ops) with keep\_prob is deprecated and will be removed in a future versio

n.

Instructions for updating:

Please use `rate` instead of `keep\_prob`. Rate should be set to `rate = 1 - k

eep\_prob`.

WARNING:tensorflow:From /home/lazic/udacity/CarND-Traffic-Sign-Classifier-Pro

ject/lenet.py:75: flatten (from tensorflow.python.keras.legacy\_tf\_layers.cor e) is deprecated and will be removed in a future version. Instructions for updating:

Use keras.layers.Flatten instead.

6 of 13 WARNING:tensorflow:From /home/lazic/.local/lib/python3.8/site-packages/tensor 12/3/20, 1:40 PM

traffic\_sign\_classifier\_project file:///home/lazic/udacity/CarND-Traffic-Sign-Class...![](traffic\_sign\_classifier\_project.007.png)

flow/python/keras/legacy\_tf\_layers/core.py:332: Layer.apply (from tensorflow. python.keras.engine.base\_layer\_v1) is deprecated and will be removed in a fut

ure version.

Instructions for updating:

Please use `layer.\_\_call\_\_` method instead.

WARNING:tensorflow:From /home/lazic/.local/lib/python3.8/site-packages/tensor flow/python/util/dispatch.py:201: softmax\_cross\_entropy\_with\_logits (from ten sorflow.python.ops.nn\_ops) is deprecated and will be removed in a future vers ion.

Instructions for updating:

Future major versions of TensorFlow will allow gradients to flow into the labels input on backprop by default.

See `tf.nn.softmax\_cross\_entropy\_with\_logits\_v2`.

**def** evaluate ( X\_data , y\_data , model **=**'lenet' ):

num\_examples **=** len ( X\_data )

total\_accuracy **=** 0

sess **=** tf **.** get\_default\_session ()

**for** offset **in** range ( 0, num\_examples , BATCH\_SIZE):

batch\_x , batch\_y **=** X\_data [ offset : offset **+** BATCH\_SIZE], y\_data [ offset : accuracy **=** sess **.** run ( accuracy\_operation , feed\_dict **=**{x: batch\_x ,

y: batch\_y , keep\_prob\_conv : 1.0 keep\_prob : 1.0 })

total\_accuracy **+=** ( accuracy **\*** len ( batch\_x )) **return** total\_accuracy **/** num\_examples

**def** predict\_single\_label ( x\_image ):

sess **=** tf **.** get\_default\_session ()

logits\_output **=** sess **.** run ( tf **.** argmax ( logits , 1),

feed\_dict **=**{

x: np**.** expand\_dims ( x\_image , axis **=**0), keep\_prob\_conv : 1.0 , keep\_prob : 1.0 })

classification\_index **=** logits\_output [ 0]

logits\_return **=** logits\_output **.** copy ()

**return** image\_labels [ classification\_index ], classification\_index , logits\_return **def** batch\_predict ( X\_data , BATCH\_SIZE**=**64):

num\_examples **=** len ( X\_data )

batch\_predict **=** np**.** zeros ( num\_examples , dtype **=**np**.** int32 )

sess **=** tf **.** get\_default\_session ()

**for** offset **in** range ( 0, num\_examples , BATCH\_SIZE):

batch\_x **=** X\_data [ offset : offset **+** BATCH\_SIZE]

batch\_predict [ offset : offset **+** BATCH\_SIZE] **=** sess **.** run ( tf **.** argmax ( logits

keep\_pr

**return** batch\_predict

saver **=** tf **.** train **.** Saver ()

**with** tf **.** Session ( config **=**tf **.** ConfigProto ( log\_device\_placement **=True** )) **as** sess :

sess **.** run ( tf **.** global\_variables\_initializer ())

num\_examples **=** len ( X\_train )

print ( "Training..." )

print ()

**for** i **in** range ( EPOCHS):

X\_train , y\_train **=** shuffle ( X\_train , y\_train )

**for** offset **in** range ( 0, num\_examples , BATCH\_SIZE):

end **=** offset **+** BATCH\_SIZE

batch\_x , batch\_y **=** X\_train [ offset : end ], y\_train [ offset : end ] sess **.** run ( training\_operation , feed\_dict **=**{x: batch\_x ,

y: batch\_y , keep\_prob\_conv : 1.0 , keep\_prob : 0.7 })

print ( "EPOCH  {} ..." **.** format ( i **+** 1))

validation\_accuracy **=** evaluate ( X\_valid , y\_valid )

print ( "Validation Accuracy =  {:.3f} "**.** format ( validation\_accuracy )) print ()

saver **.** save ( sess , './model/lenet' ) print ( "Model saved" )

Device mapping:

/job:localhost/replica:0/task:0/device:XLA\_CPU:0 -> device: XLA\_CPU device /job:localhost/replica:0/task:0/device:XLA\_GPU:0 -> device: XLA\_GPU device

Training...

EPOCH 1 ...

Validation Accuracy = 0.680

EPOCH 2 ...

Validation Accuracy = 0.824

EPOCH 3 ...

Validation Accuracy = 0.855

EPOCH 4 ...

Validation Accuracy = 0.901

EPOCH 5 ...

Validation Accuracy = 0.913

EPOCH 6 ...

Validation Accuracy = 0.899

EPOCH 7 ...

Validation Accuracy = 0.928

EPOCH 8 ...

Validation Accuracy = 0.922

EPOCH 9 ...

Validation Accuracy = 0.927

EPOCH 10 ...

Validation Accuracy = 0.917

EPOCH 11 ...

Validation Accuracy = 0.922

EPOCH 12 ...

Validation Accuracy = 0.927

EPOCH 13 ...

Validation Accuracy = 0.930

EPOCH 14 ...

Validation Accuracy = 0.932

EPOCH 15 ...

Validation Accuracy = 0.941

EPOCH 16 ...

Validation Accuracy = 0.934

EPOCH 17 ...

Validation Accuracy = 0.943

EPOCH 18 ...

Validation Accuracy = 0.941

EPOCH 19 ...

Validation Accuracy = 0.944

EPOCH 20 ...

Validation Accuracy = 0.938

EPOCH 21 ...

Validation Accuracy = 0.936

EPOCH 22 ...

Validation Accuracy = 0.934

EPOCH 23 ...

Validation Accuracy = 0.941

EPOCH 24 ...

Validation Accuracy = 0.934

EPOCH 25 ...

Validation Accuracy = 0.932

EPOCH 26 ...

Validation Accuracy = 0.937

EPOCH 27 ...

Validation Accuracy = 0.945

EPOCH 28 ...

Validation Accuracy = 0.944

EPOCH 29 ...

Validation Accuracy = 0.938

EPOCH 30 ...

Validation Accuracy = 0.947

- Check Test Accuracy

**with** tf **.** Session () **as** sess :

saver **.** restore ( sess , tf **.** train **.** latest\_checkpoint ( './model/.' )) test\_accuracy **=** evaluate ( X\_test , y\_test )

print ( "Test Accuracy =  {:.3f} "**.** format ( test\_accuracy )) INFO:tensorflow:Restoring parameters from ./model/./lenet

Test Accuracy = 0.934

Model testing

In the cells below I have addressed the rubric points regarding testing the model on new images.

'Acquiring New Images' : The model was tested on 5 new images of German Traffic signs found on the web and they are diplayed in the cell below. The images are converted to grayscale and resized to (32, 32, 1) size in order to be compatible with the trained model.

'Performance on New Images' : The model is evaluated on the new images found on the web. The results are displayed as a batch prediction and as single image. The model was able to recognize the new signs.

'Model Certainty - Softmax Probabilities' :

**import** matplotlib.image **as** mpimg

sign\_1 **=** './new\_images/nopassing.png' # label : 9 sign\_2 **=** './new\_images/keepleft.jpg' # label : 39 sign\_3 **=** './new\_images/70.jpg' # label : 4

sign\_4 **=** './new\_images/yield.png' # label : 13 sign\_5 **=** './new\_images/turnleft.jpg' # label : 33

new\_signs **=** [ sign\_1 , sign\_2 , sign\_3 , sign\_4 , sign\_5 ] sign\_images **=** []

new\_images\_label **=** [ 9, 39, 4, 13, 34]

dst\_size **=** ( 32, 32)

**for** sign **in** new\_signs :

image **=** cv2 **.** imread ( sign )

image **=** cv2 **.** resize ( image , dst\_size ) sign\_images **.** append ( image )

fig **=** plt **.** figure ( figsize **=**( 150 , 200 ))

ax **=** []

print ( 'Original images' )

**for** i **in** range ( len ( sign\_images )):

ax **.** append ( fig **.** add\_subplot ( 32, 32, i **+**1))

plt **.** imshow ( cv2 **.** cvtColor ( sign\_images [ i ], cv2 **.** COLOR\_BGR2RGB)) images\_grayscale **=** dataset\_grayscale ( sign\_images )

Original images

- Check bluk accuracy of new traffic signs

**with** tf **.** Session () **as** sess :

saver **.** restore ( sess , tf **.** train **.** latest\_checkpoint ( './model/.' )) test\_accuracy **=** evaluate ( images\_grayscale , new\_images\_label )

print ( "Test Accuracy =  {:.3f} "**.** format ( test\_accuracy ))

print ( ' \n\n Individual signs detection: \n\n ' )

- Check individual accuracy of new traffic signs

predicted\_images **=** []

**with** tf **.** Session () **as** sess :

saver **.** restore ( sess , tf **.** train **.** latest\_checkpoint ( './model/.' ))

**for** image **in** images\_grayscale :

predicted\_image , predicted\_label , logits\_result **=** predict\_single\_label print ( "Sign: " **+** predicted\_image **+** ", label : " **+** str ( predicted\_label predicted\_images **.** append ( predicted\_image )

INFO:tensorflow:Restoring parameters from ./model/./lenet Test Accuracy = 1.000

Individual signs detection:

INFO:tensorflow:Restoring parameters from ./model/./lenet

Sign: No passing, label : 9

Sign: Keep left, label : 39

Sign: Speed limit (70km/h), label : 4

Sign: Yield, label : 13

Sign: Turn left ahead, label : 34

top\_k **=** tf **.** nn**.** top\_k ( logits , k**=**5)

**def** top\_five\_outputs ( x\_image ):

sess **=** tf **.** get\_default\_session ()

top\_k\_output **=** sess **.** run ( top\_k ,

feed\_dict **=**{

x: np**.** expand\_dims ( x\_image , axis **=**0), keep\_prob\_conv : 1.0 , keep\_prob : 1.0 })

**return** top\_k\_output

**with** tf **.** Session () **as** sess :

saver **.** restore ( sess , tf **.** train **.** latest\_checkpoint ( './model/.' ))

**for** i **in** range ( len ( images\_grayscale )):

top\_five **=** top\_five\_outputs ( images\_grayscale [ i ])

print ( ' \n For predicted image : ' **+** predicted\_images [ i ] **+** ' the models top five **for** j **in** range ( 5):

label **=** top\_five [ 1][ 0][ j ]

probability **=** str ( top\_five [ 0][ 0][ j ])

print ( "Label: " **+** image\_labels [ label ] **+** "  \n Probability of: " **+** pro INFO:tensorflow:Restoring parameters from ./model/./lenet

For predicted image : No passing the models top five probabilities are: Label: No passing 

Probability of: 64.2387%

Label: No passing for vehicles over 3.5 metric tons Probability of: 23.424202%

Label: Slippery road Probability of: 18.95478%

Label: No vehicles Probability of: 14.085436%

Label: Keep right Probability of: 13.007518%

For predicted image : Keep left the models top five probabilities are: Label: Keep left 

Probability of: 44.58224%

Label: Speed limit (30km/h) Probability of: 21.253754%

Label: Wild animals crossing 

Probability of: 13.622006%

Label: Go straight or left Probability of: 12.7415695%

Label: Children crossing Probability of: 10.602471%

For predicted image : Speed limit (70km/h) the models top five probabilities are: 

Label: Speed limit (70km/h) 

Probability of: 10.596546%

Label: Road work Probability of: 6.823794%

Label: Go straight or right Probability of: 5.826931%

Label: Stop 

Probability of: 3.0191207%

Label: Dangerous curve to the left Probability of: 2.5948958%

For predicted image : Yield the models top five probabilities are: Label: Yield 

Probability of: 69.96893%

Label: Priority road Probability of: 26.387527%

Label: Speed limit (30km/h) Probability of: 22.719868%

Label: Road work Probability of: 19.574568%

Label: No vehicles Probability of: 14.047288%

For predicted image : Turn left ahead the models top five probabilities are: Label: Turn left ahead 

Probability of: 31.691187%

Label: Ahead only Probability of: 8.355588%

Label: Speed limit (60km/h) Probability of: 5.421611%

Label: Keep right Probability of: 4.8690567%

Label: Bicycles crossing Probability of: -0.2695939%
PAGE17 of NUMPAGES17 12/3/20, 1:40 PM
