# Convolutional Neural Networks for Visual Recognition (Spring 2019)

<!-- scp /home/chenxu.cnarutox/cs231n/index.html root@66.42.107.116:/root/chenwx.com/other/cs231n -->

### Lecture 1 | Introduction to CNN for Visual Recognition

---

### Lecture 2 | Image Classification

- k-Nearest Neighbor

  > on images never used

  - L1 (Manhattan) distance: $d_1(I_1,I_2)=\sum_p{|I_1^p-I_2^p|}$
  - L2 (Euclidean) distance: $d_1(I_1,I_2)=\sqrt{\sum_p{(I_1^p-I_2^p)^2}}$

- Setting Hyperparameters
  1. Split data into train, validation, and test (only used once)
  2. Cross-Validation: Split data into folds
- Linear Classification
  - $s=f(x,W) = Wx + b$

---

### Lecture 3 | Loss Functions and Optimization

- Multiclass SVM
  - $L_i=\sum_{j\not =y_i}\max(0,f(x_i;W)_j-f(x_i;W)_{y_i}+1)$
  - Loss Function:
    $$L=\frac{1}{N}\sum_{i=1}^NL_i=\frac{1}{N}\sum_{j\not =y_i}\max(0,f(x_i;W)_j-f(x_i;W)_{y_i}+1)$$
- Regularization
  - $$L(W)=\frac{1}{N}\sum_{i=1}^NL_i(f(x_i,W),y_i)+\lambda{R(W)}$$
  - L2 regularization: $R(W)=\sum_k\sum_lW_{k,l}^2$
  - L1 regularization: $R(W)=\sum_k\sum_l|W_{k,l}|$
  - Elastic net (L1 + L2): $R(W)=\sum_k\sum_l\beta{W_{k,l}^2}+|W_{k,l}|$
  - Dropout, Batch normalization, Stochastic depth, fractional pooling, etc
- Softmax Classifier (Multinomial Logistic Regression)
  - Softmax Function: $P(Y=k|X=x_i)=\frac{e^{s_k}}{\sum_j{e^{s_j}}}$
  - $L_i=-log(\frac{e^{s_k}}{\sum_j{e^{s_j}}})$
  - Loss Function:
    $$L=\frac{1}{N}\sum_{i=1}^NL_i=-\frac{1}{N}log(\frac{e^{s_k}}{\sum_j{e^{s_j}}})$$
- Stochastic Gradient Descent (SGD) = On-line Gradient Descent
- Image Features

### Lecture 4 | Introduction to Neural Networks

- Computational graphs
  - Patterns in Gradient Flow
    - Max Gate: gradient router
- ![Snipaste_2020-02-23_16-20-44](/assets/Snipaste_2020-02-23_16-20-44_p5tnag4t1.png)

<!-- sdas < img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">sasdsd -->
