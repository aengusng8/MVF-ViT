#Đây là để convolution thành thành 1 tensor 1*D
import torch
import torch.nn as nn


########################################################################
class CNN_to_Tensors(nn.Module):
    def __init__(self):
        super(CNN_to_Tensors, self).__init__()
        self.conv1 = nn.Conv2d(3, 1024, kernel_size=3, stride=1, padding=1) #(1, 1024, 70, 70)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # giả dụ mình đang có một tensor 4 chiều (1, 1024, 70, 70) -> sau khi qua layer này nó sẽ là ( 1,1024,1,1) = (1,1024)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return x


model = CNN_to_Tensors()

input_tensor = torch.randn(1, 3, 900, 800) # batch_size , channel , width , length
                                           # anh thay bằng tensor của ảnh cropped nhé 


output_tensor = model(input_tensor)

# Check the shape of the output tensor
print(output_tensor.shape)  # Output: torch.Size([1, 1024])












########################################################################
#Đây là classifier
import torch
import torch.nn as nn

class LocalClassifier(nn.Module):
    def __init__(self, token_dim):
        super(LocalClassifier, self).__init__()
        self.left_eye_layer = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.BatchNorm1d(token_dim // 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(token_dim // 2 , token_dim // 4),
            nn.BatchNorm1d(token_dim // 4),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.right_eye_layer = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.BatchNorm1d(token_dim // 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(token_dim // 2 , token_dim // 4),
            nn.BatchNorm1d(token_dim // 4),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.nose_layer = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.BatchNorm1d(token_dim // 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(token_dim // 2 , token_dim // 4),
            nn.BatchNorm1d(token_dim // 4),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.mouth_layer = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.BatchNorm1d(token_dim // 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(token_dim // 2 , token_dim // 4),
            nn.BatchNorm1d(token_dim // 4),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(token_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, left_eye_input, right_eye_input, nose_input, mouth_input):
        left_eye_layer = self.left_eye_layer(left_eye_input)
        right_eye_layer = self.right_eye_layer(right_eye_input)
        nose_layer = self.nose_layer(nose_input)
        mouth_layer = self.mouth_layer(mouth_input)
        concatenated = torch.cat((left_eye_layer, right_eye_layer, nose_layer, mouth_layer), dim=1)
        # concatenated = concatenated.flatten()
        # print("shape of concat : " + str(concatenated.shape))
        output = self.output_layer(concatenated)
        output = self.sigmoid(output)
        return output

D = 1024
model = LocalClassifier(D)

# Define the loss function
criterion = nn.BCELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



########################################################################
#Đây là gen ra Q-local
import torch
import torch.nn as nn

class Qlocal(nn.Module):
    def __init__(self, token_dim = 1024):
        super(Qlocal, self).__init__()
        self.token_dim = token_dim
        # self.conv = nn.Sequential(
        #     nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True)
        # )
        # self.fc = nn.Linear(in_features=4*token_dim, out_features=16*token_dim) # uncomment nếu muốn dùng convo trở lại 
        self.fc = nn.Linear(in_features=token_dim, out_features=4*token_dim)
        
    def forward(self, x):
        x = torch.cat(x, dim=0).view(4, self.token_dim)
        # x = self.conv(x.unsqueeze(0))
        # x = x.view(-1, 4*self.token_dim)
        x = self.fc(x)
        x = x.view(-1, self.token_dim)
        return x
token_dim = 1024
Q_gen = Qlocal(token_dim)

# Create input tensors
t1 = torch.randn(token_dim)
t2 = torch.randn(token_dim)
t3 = torch.randn(token_dim)
t4 = torch.randn(token_dim)

# Extract features
Q = Q_gen([t1, t2, t3, t4])
print(Q.shape) # Should output (16, 1024)

########################################################################
# Test cho classifier : 
# Generate some sample data
N = 1000
D = 1024
x1 = torch.randn(N, D)
x2 = torch.randn(N, D)
x3 = torch.randn(N, D)
x4 = torch.randn(N, D)
y = torch.randint(0, 2, (N, 1)).float()

# Split the data into train and test sets
x1_train, x1_test = x1[:800], x1[800:]
x2_train, x2_test = x2[:800], x2[800:]
x3_train, x3_test = x3[:800], x3[800:]
x4_train, x4_test = x4[:800], x4[800:]
y_train, y_test = y[:800], y[800:]

# Train the model
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x1_train, x2_train, x3_train, x4_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

# Evaluate the model
with torch.no_grad():
    y_pred = model(x1_test, x2_test, x3_test, x4_test)
    loss = criterion(y_pred, y_test)
    accuracy = ((y_pred > 0.5) == y_test).sum().item() / len(y_test)

print('Accuracy:', accuracy)



