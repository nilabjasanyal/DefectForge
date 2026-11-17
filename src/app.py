
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from evaluate_sam import segment_defect


def analyze_image(image):

    if image is None:
        raise gr.Error("Please upload a screw image.")

    # Run DefectForge
    predicted_mask, result, anomaly_map = segment_defect(image)

    original = image.convert("RGB").resize((518, 518))
    image_array = np.array(original)

    # Create normalized heatmap for display
    heatmap = anomaly_map - anomaly_map.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)

    colors = plt.get_cmap("jet")(heatmap)[:, :, :3]
    colors = (colors * 255).astype(np.uint8)

    heatmap_overlay = (
        0.6 * image_array + 0.4 * colors
    ).astype(np.uint8)

    # Create red segmentation overlay
    segmentation_overlay = image_array.copy()

    segmentation_overlay[predicted_mask] = (
        0.5 * segmentation_overlay[predicted_mask]
        + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    details = (
        f"Selected anomaly peak: {result['peak']}\n"
        f"Selected SAM mask: {result['mask_number']}\n"
        f"Combined ranking score: {result['combined_score']:.4f}"
    )

    return (
        Image.fromarray(heatmap_overlay),
        Image.fromarray(segmentation_overlay),
        details
    )


with gr.Blocks(title="DefectForge") as demo:

    gr.Markdown(
        "# DefectForge\n"
        "### Few-Shot Industrial Anomaly Detection\n"
        "Powered by DINOv2 + FAISS + Segment Anything"
    )

    with gr.Row():
        input_image = gr.Image(
            label="Upload Screw Image",
            type="pil"
        )

    analyze_button = gr.Button(
        "Analyze Image",
        variant="primary"
    )

    with gr.Row():
        heatmap_output = gr.Image(
            label="Anomaly Heatmap"
        )

        segmentation_output = gr.Image(
            label="SAM Defect Segmentation"
        )

    details_output = gr.Textbox(
        label="Analysis Details",
        lines=3
    )

    analyze_button.click(
        fn=analyze_image,
        inputs=input_image,
        outputs=[
            heatmap_output,
            segmentation_output,
            details_output
        ]
    )


if __name__ == "__main__":
    demo.launch()
