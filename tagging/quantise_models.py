import argparse
from pathlib import Path
from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer
from optimum.pipelines import pipeline
import onnxruntime as ort


def convert_and_quantize_model(model_path, onnx_suffix, quantized_suffix):
    # Define paths for ONNX and quantized models
    onnx_path = Path(model_path).parent / f"{onnx_suffix}_onnx"
    quantized_path = Path(model_path).parent / f"{quantized_suffix}_quantised"

    # Step 1: Load the model and tokenizer, converting to ONNX format
    print("Loading and converting model to ONNX with limited threading...")

    # Set ONNX Runtime SessionOptions to limit threads
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    model = ORTModelForTokenClassification.from_pretrained(model_path, from_transformers=True,
                                                           session_options=session_options)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Save the ONNX model and tokenizer
    model.save_pretrained(onnx_path)
    tokenizer.save_pretrained(onnx_path)
    print(f"ONNX model and tokenizer saved at {onnx_path}")

    # Step 2: Quantize the ONNX model
    print("Quantizing ONNX model...")
    quantizer = ORTQuantizer.from_pretrained(onnx_path)
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    # Quantize and save the quantized model
    model_quantized_path = quantizer.quantize(
        save_dir=quantized_path,
        quantization_config=dqconfig,
    )
    print(f"Quantized model saved at: {model_quantized_path}")

    # Test the quantized model
    test_quantized_model(quantized_path, session_options)


def test_quantized_model(quantized_path, session_options):
    # Load the quantized model and tokenizer with limited threading
    print("Loading quantized model for testing...")
    model_quantized = ORTModelForTokenClassification.from_pretrained(
        quantized_path, file_name="model_quantized.onnx", session_options=session_options
    )
    tokenizer_quantized = AutoTokenizer.from_pretrained(quantized_path, model_max_length=512, batch_size=8,
                                                        truncation=True)

    # Initialize the pipeline with the quantized model
    ner_quantized = pipeline("token-classification", model=model_quantized, tokenizer=tokenizer_quantized,
                             aggregation_strategy="max")

    # Sample text for testing
    text = '''Chronic kidney disease (CKD) is a global public health problem, and its prevalence is gradually increasing,
              mainly due to an increase in the number of patients with type 2 diabetes mellitus (T2DM). Human multidrug
              and toxin extrusion member 2 (MATE2-K, SLC47A2) plays an important role in the renal elimination of various
              clinical drugs including the antidiabetic drug metformin.'''

    # Run the quantized model on sample text and print the results
    print("Running quantized model on sample text...")
    results = ner_quantized(text)
    for entity in results:
        print(f" - Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']}")


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert and quantize a model to ONNX format with custom suffixes.")
    parser.add_argument("model_path", type=str, help="Path to the pretrained model directory")
    parser.add_argument("suffix", type=str, help="Suffix for naming ONNX and quantized directories")

    # Parse arguments
    args = parser.parse_args()

    # Run the conversion and quantization with command-line arguments
    convert_and_quantize_model(args.model_path, args.suffix, args.suffix)
