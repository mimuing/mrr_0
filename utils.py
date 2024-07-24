import os
import json

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))

def save_results(results, path):
    with open(path, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    # 예시 사용
    model = get_model()
    save_model(model, 'model.pth')
