## Bike Rental Ridership Prediction

---
#### demo

learning curves

![learning_curve](https://github.com/ZhangShiqiu1993/deep_learning/blob/master/bike_rental_ridership/learning_curve.png?raw=true)

prediction

![prediction](https://github.com/ZhangShiqiu1993/deep_learning/blob/master/bike_rental_ridership/prediction.png?raw=true)

---

+ implemented three layers nerual network model; implement forward propagation and backpropagation
+ trained NN model based on dataset from Capital BikeShare System
+ implemented daily bike rental ridership prediction

---

+ 基于Python实现三层神经网络；使用sigmoid函数作为激活函数；实现forward propagation和Backpropagation算法。
+ 基于Capital BikeShare system 2011-2012数据集，训练神经网络，实现每日共享单车租借预测。

---

*Suggestion from my grader on choosing hyper parameters*
> We should be able to increase the loss on the validation set further with more epochs and a smaller learning rate, while still staying time efficient.

>The problem of smaller neural nets is that they have trouble generalizing. Since it has less parameters it can overfit the data you have and perform worse in real life.

>Of course more doesn't mean better, there is a optimum point in the middle but 6 was too low.

>The quotient of the learning rate / the number of records should end up around 0.01 (i.e. self.lr / n_records ~ 0.01), although, much like with the number of hidden units, it is results that matter as far as meeting specifications is concerned. (The network converges in a reasonable amount of time.)

>Sometimes, the network doesn't converge when that quotient is around or above 0.1. The weight update steps are too large with this learning rate and the weights end up not converging.

>On the other side, should be using learning rates that are just low enough to get the network to converge, there really isn't a benefit in a really low learning rate. I'd say to keep that quotient larger than about 0.001.
