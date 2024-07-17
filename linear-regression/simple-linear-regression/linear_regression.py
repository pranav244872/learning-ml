import matplotlib.pyplot as plt
import numpy as np
#Applying linear regression with full formula so i understand what the hell is going on
#Dataset
x_values = [1.5, 2.0, 1.8, 2.2, 1.7, 1.9, 2.1, 2.3]
y_values = [250, 320, 280, 350, 270, 300, 330, 380]

#Defining a function to find the Cost function for a given value of w and b:
# def cost_function(w,b):
#     cost_function = 0
#     for i in range(len(x_values)):
#         cost_function += ((w*x_values[i]+b)-y_values[i])**2
#     cost_function = (1/(2*len(x_values)))*cost_function
#     return cost_function


def cost_function(w,b):
    d=[]
    v=[]
    mv=0
    for i in range(len(x_values)):
        d+=((w*x_values[i]+b)-y_values[i])
    for value in d:
        v+=((d-mean(d))**2)/len(d)
    mv+=mean(v)     


#Finding the cost function for a random value of w = 1 and b = 1
print(f'The cost function for w=1 and b=1 is {cost_function(1,1)}')

#Defining the gradient_descent formula to reduce the values of w and b simultaneously and iteratively
def gradient_descent(w,b,alpha,iterations):

    #Finding the partial derivative of the cost function with respect to w
    def partial_dervw(w,b):
        partial_dervw= 0
        for i in range(len(x_values)):
            partial_dervw += ((w*x_values[i]+b)-y_values[i])*x_values[i]
        partial_dervw *= 1/len(x_values)
        return partial_dervw

    #Finding the partial derivative of the cost function with respect to b
    def partial_dervb(w,b):
        partial_dervb= 0
        for i in range(len(x_values)):
            partial_dervb += ((w*x_values[i]+b)-y_values[i])
        partial_dervb *= 1/len(x_values)
        return partial_dervb

    for i in range(iterations):
        #Formulas for gradient_descent
        w_tmp = w - alpha*partial_dervw(w, b)
        b_tmp = b - alpha*partial_dervb(w, b)
        w = w_tmp
        b = b_tmp

    print(f'w = {w} b = {b}')
   
    #Show the graph for the current gradient_descent
    x = np.linspace(0,10,100)
    y = w*x + b
    plt.scatter(x_values,y_values,color='blue')
    plt.plot(x,y,color='red')
    plt.show()

#Try this with different values of alpha and iterations to find perfect fit
#you always wont get correct answer if you dont select optimal alpha and iteration values
gradient_descent(1, 1, 0.1,10000)
