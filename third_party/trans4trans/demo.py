import logging
import os
import sys
import torch
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path


cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg



def find_connected_components_graph(adjacency_list, nodes):
    """
    Finds connected components from an adjacency-list graph with DFS.
    """
    visited = set()
    components = []
    for node in nodes:
        if node not in visited:
            component = []
            stack = [node]
            visited.add(node)
            while stack:
                current_node = stack.pop()
                component.append(current_node)
                if current_node in adjacency_list:
                    for neighbor in adjacency_list[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
            components.append(component)
    return components



def demo():
    output_dir = 'output'
    for i, arg in enumerate(sys.argv):
        if arg == '--output-dir' and i+1 < len(sys.argv):
            output_dir = sys.argv[i+1]
            sys.argv.pop(i+1)
            sys.argv.pop(i)
            break

    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    # ==================================================================

    # ==================================================================
    output_base_dir = Path(output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output will be saved to: {output_base_dir}")


    MIN_OBJECT_AREA = 1000

    INTER_CLASS_DILATION = 0

    INTRA_CLASS_DILATION = 0


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])
    model = get_segmentation_model().to(args.device)
    model.eval()

    if os.path.isdir(args.input_img):
        img_paths = [os.path.join(args.input_img, x) for x in os.listdir(args.input_img)]
    else:
        img_paths = [args.input_img]

    for img_path in img_paths:
        logging.info(f"\nProcessing image: {img_path}")
        image = Image.open(img_path).convert('RGB')
        image_resized = image.resize((512, 512))
        images = transform(image_resized).unsqueeze(0).to(args.device)

        with torch.no_grad():
            output = model(images)

        pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()

        # ==================================================================

        # ==================================================================
        img_basename = Path(img_path).stem
        image_output_dir = output_base_dir / img_basename
        image_output_dir.mkdir(exist_ok=True)


        detected_class_ids = np.unique(pred)
        object_class_ids = [cid for cid in detected_class_ids if cid != 0]



        class_instances = defaultdict(list)
        for cid in object_class_ids:
            class_mask = (pred == cid).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= MIN_OBJECT_AREA:
                    instance_mask = (labels == i)
                    class_instances[cid].append(instance_mask)

        if not class_instances:
            logging.info("No objects remained after filtering by area.")
            continue


        logging.info("Step 1: Merging close instances within the same class (distance < 50)...")
        intra_class_kernel = np.ones((INTRA_CLASS_DILATION, INTRA_CLASS_DILATION), np.uint8)




        aggregated_blobs = []

        for cid, instances in class_instances.items():
            if not instances: continue


            adjacency_list = defaultdict(set)
            for i in range(len(instances)):
                for j in range(i + 1, len(instances)):
                    dilated_i = cv2.dilate(instances[i].astype(np.uint8), intra_class_kernel).astype(bool)

                    if np.any(np.logical_and(dilated_i, instances[j])):
                        adjacency_list[i].add(j)
                        adjacency_list[j].add(i)


            instance_indices = list(range(len(instances)))
            connected_groups = find_connected_components_graph(adjacency_list, instance_indices)


            for group in connected_groups:
                merged_mask = np.zeros_like(pred, dtype=bool)
                for idx in group:
                    merged_mask = np.logical_or(merged_mask, instances[idx])


                aggregated_blobs.append({"mask": merged_mask, "original_cid": cid})


        logging.info("Step 2: Merging close blobs from different classes (distance < 10)...")
        if not aggregated_blobs: continue

        inter_class_kernel = np.ones((INTER_CLASS_DILATION, INTER_CLASS_DILATION), np.uint8)


        blob_adjacency_list = defaultdict(set)
        for i in range(len(aggregated_blobs)):
            for j in range(i + 1, len(aggregated_blobs)):
                blob1 = aggregated_blobs[i]
                blob2 = aggregated_blobs[j]


                if blob1["original_cid"] != blob2["original_cid"]:
                    dilated_mask1 = cv2.dilate(blob1["mask"].astype(np.uint8), inter_class_kernel).astype(bool)
                    if np.any(np.logical_and(dilated_mask1, blob2["mask"])):
                        blob_adjacency_list[i].add(j)
                        blob_adjacency_list[j].add(i)


        blob_indices = list(range(len(aggregated_blobs)))
        final_groups = find_connected_components_graph(blob_adjacency_list, blob_indices)

        logging.info(f"Generated {len(final_groups)} final mask(s).")


        for i, group in enumerate(final_groups):
            final_mask_np = np.zeros_like(pred, dtype=bool)
            for idx in group:
                final_mask_np = np.logical_or(final_mask_np, aggregated_blobs[idx]["mask"])

            final_mask_img = Image.fromarray(final_mask_np.astype(np.uint8) * 255)

            mask_filename = f"mask_{i}.png"
            save_path = image_output_dir / mask_filename
            final_mask_img.save(save_path)
            logging.info(f"Saved final mask {i} to {save_path}")


if __name__ == '__main__':
    demo()
