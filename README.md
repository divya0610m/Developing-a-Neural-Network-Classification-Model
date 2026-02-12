# Experiment 2 : Developing a Neural Network Classification Model
## NAME : DIVYA LAKSHMI M
## REGISTRATION NUMBER : 212224040082

## AIM :
To develop a neural network classification model for the given dataset.

## THEORY :
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model:
![WhatsApp Image 2026-02-10 at 11 43 55 AM (1)](https://github.com/user-attachments/assets/dd5f490b-4ea3-4abe-9eb0-496ce50e2dce)

## DESIGN STEPS
### STEP 1: 
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2: 
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.


### STEP 3: 
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.


### STEP 4: 

Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5: 

Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6: 
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.




## PROGRAM:

### Name: DIVYA LAKSHMI M

### Register Number: 212224040082

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)


    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs,labels in train_loader:
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

train_model(model,train_loader,criterion,optimizer,epochs=100)
```

### Dataset Information
<img width="1293" height="290" alt="Screenshot 2026-02-10 112442" src="https://github.com/user-attachments/assets/8b32ab76-dda0-4724-ab9a-335d350ab0d9" />


### OUTPUT
<img width="460" height="56" alt="output exp2" src="https://github.com/user-attachments/assets/32729937-1ad8-42b5-8a12-de6080e5e7fc" />


## Confusion Matrix
<img width="699" height="592" alt="Screenshot 2026-02-10 112411" src="https://github.com/user-attachments/assets/debce90a-6522-4e72-9c08-981fd0ae3502" />


## Classification Report
<img width="577" height="435" alt="Screenshot 2026-02-10 112420" src="https://github.com/user-attachments/assets/69a0c335-7555-45f5-84eb-359c8551c1fb" />


### New Sample Data Prediction
<img width="501" height="128" alt="Screenshot 2026-02-10 112359" src="https://github.com/user-attachments/assets/5e292e77-3fa3-463a-b4a8-4f1913dae290" />


## RESULT
This program has been executed successfully.
