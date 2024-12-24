import gradio as gr
import onnx
from collections import Counter
import zipfile
import os

def process_onnx(uploaded_file):
    # Check if the uploaded file is a zip file
    if zipfile.is_zipfile(uploaded_file.name):
        with zipfile.ZipFile(uploaded_file.name, 'r') as zip_ref:
            zip_ref.extractall("/tmp")
            onnx_file = zip_ref.namelist()[0]  # Assuming there is only one ONNX file in the zip
            file_path = os.path.join("/tmp", onnx_file)
    else:
        file_path = uploaded_file.name

    # Load the ONNX model
    model = onnx.load(file_path)

    # Collect basic information about the model
    info = {
        "Model Name": model.graph.name,
        "Number of Nodes": len(model.graph.node),
        "Architecture Summary": Counter(),
        "Nodes": []
    }

    # Iterate over each node (layer) to collect detailed information
    for node in model.graph.node:
        node_info = {
            "Name": node.name,
            "Type": node.op_type,
            "Inputs": node.input,
            "Outputs": node.output
        }
        info["Nodes"].append(node_info)
        info["Architecture Summary"][node.op_type] += 1

    # Format the summary output
    summary_output = '\n'.join([f"{key}: {value}" for key, value in info.items() if key != "Nodes"])

    # Format the complete nodes output
    nodes_output = "Complete Nodes:\n" + '\n'.join([str(node) for node in info["Nodes"]])

    return summary_output, nodes_output

# Define the Gradio Interface
iface = gr.Interface(
    fn=process_onnx,
    inputs=gr.File(label="Upload .ONNX or .ZIP File"),
    outputs=[
        gr.components.Textbox(label="Summary"),
        gr.components.Textbox(label="Complete Nodes")
    ],
    examples=["example1.onnx"],  # Add your example file here
    title="ONNX Model Scope",
    description="Upload an ONNX file or a ZIP file containing an .onnx file to extract and display its detailed information. This process can take some time depending on the size of the ONNX model."
)

# Launch the Interface
iface.launch(debug=True)
