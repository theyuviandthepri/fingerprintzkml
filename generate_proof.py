import ezkl
import os
import asyncio
import psutil
import json


# ==================================================
# Configuration & Memory Management
# ==================================================
def check_available_memory():
    """Check available system memory in GB"""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024 ** 3)
    total_gb = memory.total / (1024 ** 3)
    print(f"💾 Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
    return available_gb


def print_memory_status():
    """Print current memory usage"""
    memory = psutil.virtual_memory()
    used_percent = memory.percent
    print(f"📊 Memory Usage: {used_percent:.1f}%")
    if used_percent > 80:
        print("⚠️  WARNING: Memory usage is high!")
    return used_percent


# ==================================================
# File Paths
# ==================================================
os.makedirs("zkml_data", exist_ok=True)

model_path = os.path.join("models", "siamese_fingerprint_zkml.onnx")
data_path = os.path.join("models", "input.json")

settings_path = os.path.join("zkml_data", "settings.json")
compiled_model_path = os.path.join("zkml_data", "network.compiled")

pk_path = os.path.join("zkml_data", "proving.key")
vk_path = os.path.join("zkml_data", "verification.key")

proof_path = os.path.join("zkml_data", "proof.json")
witness_path = os.path.join("zkml_data", "witness.json")

srs_path = os.path.join("zkml_data", "kzg.srs")


# ==================================================
# Async Helper with Error Handling
# ==================================================
async def execute_ezkl(func, *args, **kwargs):
    """Execute EZKL function with error handling"""
    try:
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


# ==================================================
# Main Pipeline (Optimized)
# ==================================================
async def run_zkml_pipeline():

    print("\n🚀 Starting Optimized Fingerprint ZKML Pipeline\n")

    # Check memory at start
    available_memory = check_available_memory()
    if available_memory < 8:
        print("⚠️  WARNING: Less than 8GB available. Proof generation may fail.")
        print("   Consider increasing swap or closing other applications.\n")

    cleanup_files = [
        settings_path,
        compiled_model_path,
        pk_path,
        vk_path,
        proof_path,
        witness_path,
        srs_path,
    ]

    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)

    print("🧹 Old files removed.\n")

    # --------------------------------------------------
    # Generate Settings
    # --------------------------------------------------
    print("⚙️ Generating settings...")

    await execute_ezkl(
        ezkl.gen_settings,
        model_path,
        settings_path,
        py_run_args=ezkl.PyRunArgs()
    )

    print("✅ Settings generated.\n")

    # --------------------------------------------------
    # Calibrate (OPTIMIZED)
    # --------------------------------------------------
    print("📏 Calibrating model...")
    print_memory_status()

    try:
        await execute_ezkl(
        ezkl.calibrate_settings,
        data_path,
        model_path,
        settings_path,
        "resources",
        lookup_safety_margin=2,
        scales=[8],
        scale_rebase_multiplier=[1],
        max_logrows=16
    )
    except Exception as e:
        print(f"⚠️  Calibration warning: {str(e)}")
        print("   Continuing with current settings...\n")

    print("✅ Calibration complete.\n")

    # --------------------------------------------------
    # Compile
    # --------------------------------------------------
    print("🛠️ Compiling circuit...")
    print_memory_status()

    await execute_ezkl(
        ezkl.compile_circuit,
        model_path,
        compiled_model_path,
        settings_path
    )

    print("✅ Circuit compiled.\n")

    # Check compiled model size
    if os.path.exists(compiled_model_path):
        model_size_mb = os.path.getsize(compiled_model_path) / (1024 ** 2)
        print(f"📦 Compiled model size: {model_size_mb:.1f}MB\n")

    # --------------------------------------------------
    # Download SRS
    # --------------------------------------------------
    print("⬇️ Downloading SRS...")

    await execute_ezkl(
        ezkl.get_srs,
        settings_path=settings_path,
        srs_path=srs_path,
    )

    print("⏳ Waiting for SRS download...")

    previous = -1
    stable = 0

    while stable < 2:
        await asyncio.sleep(2)

        if os.path.exists(srs_path):
            size = os.path.getsize(srs_path)

            if size > 0 and size == previous:
                stable += 1
            else:
                stable = 0

            previous = size

    print("✅ SRS Ready.\n")

    if os.path.exists(srs_path):
        srs_size_mb = os.path.getsize(srs_path) / (1024 ** 2)
        print(f"📦 SRS size: {srs_size_mb:.1f}MB\n")

    # --------------------------------------------------
    # Witness
    # --------------------------------------------------
    print("🧾 Generating witness...")
    print_memory_status()

    await execute_ezkl(
        ezkl.gen_witness,
        data_path,
        compiled_model_path,
        witness_path
    )

    print("✅ Witness generated.\n")

    if os.path.exists(witness_path):
        witness_size_mb = os.path.getsize(witness_path) / (1024 ** 2)
        print(f"📦 Witness size: {witness_size_mb:.1f}MB\n")

    # --------------------------------------------------
    # Setup
    # --------------------------------------------------
    print("🔑 Generating proving/verifying keys...")
    print_memory_status()

    await execute_ezkl(
        ezkl.setup,
        compiled_model_path,
        vk_path,
        pk_path,
        witness_path=witness_path,
        srs_path=srs_path,
        disable_selector_compression=True
    )

    print("✅ Keys generated.\n")

    # --------------------------------------------------
    # Prove (OPTIMIZED for Memory)
    # --------------------------------------------------
    print("🧠 Generating Zero Knowledge Proof...")
    print("   ⏳ This may take several minutes...\n")

    print_memory_status()

    try:
        # Add timeout and memory tracking
        await asyncio.wait_for(
            execute_ezkl(
                ezkl.prove,
                witness_path,
                compiled_model_path,
                pk_path,
                proof_path,
                srs_path=srs_path,
            ),
            timeout=3600  # 1 hour timeout
        )

        print("\n✅ Proof generated.\n")

    except asyncio.TimeoutError:
        print("❌ Proof generation timed out after 1 hour")
        print("   Try reducing model size or increasing system RAM")
        return False
    except MemoryError:
        print("❌ Out of Memory during proof generation")
        print("   SOLUTIONS:")
        print("   1. Increase system RAM")
        print("   2. Reduce model input size")
        print("   3. Reduce max_logrows further (try 16)")
        print("   4. Add swap space")
        return False
    except Exception as e:
        print(f"❌ Proof generation failed: {str(e)}")
        return False

    # --------------------------------------------------
    # Verify
    # --------------------------------------------------
    print("🔍 Verifying proof...")
    print_memory_status()

    try:
        valid = await execute_ezkl(
            ezkl.verify,
            proof_path,
            settings_path,
            vk_path,
            srs_path=srs_path,
            reduced_srs=False
        )

        if valid:
            print("\n🎉 SUCCESS 🎉")
            print("Fingerprint proof verified successfully.\n")

            print("Generated Files:")
            print(f"Proof        : {proof_path}")
            print(f"Verification : {vk_path}")
            print(f"Settings     : {settings_path}")
            print(f"SRS          : {srs_path}")

            # Summary
            print("\n📊 Summary:")
            if os.path.exists(proof_path):
                proof_size_mb = os.path.getsize(proof_path) / (1024 ** 2)
                print(f"Proof size: {proof_size_mb:.2f}MB")

            return True

        else:
            print("\n❌ Verification Failed")
            return False

    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")
        return False


# ==================================================
# Entry Point with Error Handling
# ==================================================
if __name__ == "__main__":
    try:
        success = asyncio.run(run_zkml_pipeline())
        if success:
            print("\n✅ Pipeline completed successfully!")
        else:
            print("\n⚠️  Pipeline did not complete successfully")
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()