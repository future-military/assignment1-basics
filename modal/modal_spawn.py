"""Spawns a training run on the already-deployed cs336-basics-train app and
exits immediately. Prints a call ID that can be used to check status or
retrieve the result later, from any machine, independent of this process."""

import modal

f = modal.Function.from_name("cs336-basics-train", "train")
call = f.spawn(
    total_steps=20000,
    warmup_steps=1000,
    batch_size=64,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
    lr_max=3e-4,
    lr_min=3e-5,
    eval_every=200,
    checkpoint_every=2000,
)
print(f"CALL_ID={call.object_id}")
