import numpy as np

class LinearRegression:
    def __init__ (self, learning_rate= 0.01, epochs= 1000):
        self.lr= learning_rate
        self.epochs= epochs
        self.weights= None
        self.bias= None
        
    def fit(self, X, y):
        
        n_samples, n_features= X.shape
        self.weights= np.zeros(n_features)
        self.bias= 0
        
        for _ in range(self.epochs):    
            y_pred= X@self.weights+ self.bias
            error= y_pred- y
            
            self.weights-= self.lr* ((X.T@error)/n_samples)
            self.bias-= self.lr*error.mean()
            
    def predict(self,X):
        return X@self.weights+ self.bias
    
    def mse(self, X, y):
        return np.mean((self.predict(X)-y)**2)
    
if __name__ == "__main__":
    np.random.seed(42)
    X= np.random.rand(100,1)
    y= 3*X.squeeze()+ 2+ np.random.randn(100)*0.1
    
    model= LinearRegression(0.1,1000)
    model.fit(X,y)
    
    print(f"mse: {model.mse(X,y)}")
    print(f"weights: {model.weights}")    
    print(f"bias: {model.bias}")    
    