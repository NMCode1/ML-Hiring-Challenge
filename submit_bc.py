# submit_bc.py
import os
from sai_rl import SAIClient
from bc_model import BCPolicy

SAVE_DIR = "models/bc"
LABEL = "Nick BC v1 (normalized MLP)"

def main():
    meta_path = os.path.join(SAVE_DIR, "meta.json")
    weights_path = os.path.join(SAVE_DIR, "model.pt")
    assert os.path.exists(meta_path), f"Missing {meta_path}"
    assert os.path.exists(weights_path), f"Missing {weights_path}"

    # Build a runnable torch.nn.Module (NOT a state_dict)
    policy = BCPolicy(meta_path, weights_path)

    # Submit as a PyTorch model; ask SAI to convert to ONNX for portability
    sai = SAIClient(comp_id="franka-ml-hiring")
    result = sai.submit_model(
        name=LABEL,
        model=policy,
        model_type="pytorch",
        use_onnx=True,          # <-- important for compliance
        # action_function can be omitted; policy outputs final actions
    )
    print("Submitted. Server response:\n", result)

if __name__ == "__main__":
    main()
