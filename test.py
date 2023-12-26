import torch
from torch.distributions.categorical import Categorical
import math
# from torch.nn.functional as F
if __name__ == "__main__":
    # action = [1, 2, 4, 3]
    # action_probs = F.softmax(action, dim=-1)
    # print(action_probs)
    action_probs = [0.24, 0.26, 0.2, 0.3]
    action = [1, 2, 3]
    print([math.log(action_probs[i]) for i in action])
    m = Categorical(torch.tensor(action_probs))
    print(m.log_prob(torch.tensor([1, 2, 3])))
    print(m.sample())
    