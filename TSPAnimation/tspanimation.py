import os
import shutil

import numpy as np
import torch
import IPython.display
import ffmpeg

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from PIL import Image
from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTImageProcessor, ViTModel
import argparse


def create_data_model(x: np.array):
    """
    Stores the data for the problem.

    Args:
        x (np.array): Array of locations.

    Returns:
        dict: Data model containing locations, number of vehicles, and depot index.
    """
    data = {
        "locations": x,  # Locations in block units
        "num_vehicles": 1,  # Number of vehicles
        "depot": 0,  # Depot index
    }
    return data


def compute_euclidean_distance_matrix(locations: list):
    """
    Computes a distance matrix for a given list of locations using the cosine distance.

    Args:
        locations (list): A list of tuples, where each tuple represents the coordinates (x, y) of a location.

    Returns:
        dict: A nested dictionary where the keys are indices of the locations and the values are dictionaries
              with distances to other locations. The distance between a location and itself is 0.

    Note:
        The distances are scaled by a factor of 10^6 for precision.
    """
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                # distances[from_counter][to_counter] = int(
                #    math.hypot((from_node[0] - to_node[0]), (from_node[1] - to_node[1]))
                # )
                distances[from_counter][to_counter] = int(
                    10**6 * cosine(from_node, to_node)
                )
    return distances


def get_solution(
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    solution: pywrapcp.Assignment,
) -> list:
    """
    Extracts the solution from the routing model and returns the route as a list of node indices.

    Args:
        manager (pywrapcp.RoutingIndexManager): The manager that handles the conversion between node indices and routing indices.
        routing (pywrapcp.RoutingModel): The routing model used to solve the problem.
        solution (pywrapcp.Assignment): The solution obtained from the solver.

    Returns:
        list: A list of node indices representing the route in the solution.
    """
    """Prints solution on console."""
    index = routing.Start(0)
    plan_output = []
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += [
            manager.IndexToNode(index)
        ]  # f" {manager.IndexToNode(index)} ->"
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += [manager.IndexToNode(index)]  # f" {manager.IndexToNode(index)}\n"
    return plan_output


def tsp(
    x,
    time_limit=30,
    metaheuristic="GUIDED_LOCAL_SEARCH",
    initial="AUTOMATIC",
    solution_limit=None,
):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(x)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["locations"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # distance_matrix = compute_euclidean_distance_matrix(data["locations"])

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # return distance_matrix[from_node][to_node]
        # return (10**6*cosine(x[to_node],x[from_node])).astype(int)
        return (10**6 * np.linalg.norm(x[to_node] - x[from_node])).astype(int)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = eval(
        f"routing_enums_pb2.LocalSearchMetaheuristic.{metaheuristic}"
    )
    if solution_limit:
        search_parameters.solution_limit = solution_limit
    else:
        search_parameters.time_limit.seconds = time_limit

    search_parameters.first_solution_strategy = eval(
        f"routing_enums_pb2.FirstSolutionStrategy.{initial}"
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        return get_solution(manager, routing, solution)


class TSPAnimation:
    """
    A class to create an animation of images sequenced by solving the Traveling Salesman Problem (TSP)
    using Vision Transformer (ViT) features.

    Attributes:
        input_folder (str): Path to the folder containing input images.
        output_name (str): Name of the output animation file.
        batch_size (int): Number of images to process in a batch. Default is 32.
        tsp_solver_limit (int): Time limit for the TSP solver in seconds. Default is 120.
        resolution (tuple): Resolution of the output animation. Default is (2000, 2000).
        framerate (int): Frame rate of the output animation. Default is 8.
        paths (list): List of paths to the input images.
        feature_extractor (ViTImageProcessor): Pre-trained ViT feature extractor.
        model (ViTModel): Pre-trained ViT model.
        features (np.array): Extracted features of the input images.
        route (np.array): Computed TSP route of the images.

    Methods:
        extract_features():
            Extracts features from the input images using the ViT model.

        sequence_images():
            Computes the TSP route based on the extracted features.

        animate(framerate=None):
            Creates an animation of the images sequenced by the TSP route and saves it as a video and GIF.
    """

    def __init__(
        self,
        input_folder,
        output_name,
        batch_size=32,
        tsp_solver_limit=120,
        resolution=(2000, 2000),
        framerate=8,
        max_items=None,
    ):
        self.input_folder = input_folder
        self.output_name = output_name
        self.batch_size = batch_size
        self.tsp_solver_limit = tsp_solver_limit
        self.resolution = resolution
        self.framerate = framerate

        # Load ViT feature extractor
        self.feature_extractor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        # Load ViT feature extractor
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # Get individual image paths
        self.paths = [
            f"{root}/{path}"
            # Iterate directory recursively
            for root, dirs, files in os.walk(self.input_folder)
            for path in files
            # Check file extension
            if path.split(".")[-1].lower() in ["png", "jpg", "jpeg"]
        ]

        if max_items is not None:
            np.random.shuffle(self.paths)
            self.paths = self.paths[:max_items]

        paths_ = []
        for path in tqdm(self.paths, desc="> Checking images..."):
            try:
                Image.open(path)
                paths_.append(path)
            except:
                pass

        self.paths = paths_

    def extract_features(self):
        """
        Extracts features from images using a pre-trained model and a feature extractor.
        This method processes images in batches, extracts features using the specified model,
        and concatenates the features into a single numpy array.

        Attributes:
            features (np.array): A numpy array containing the extracted features.

        Args:
            None

        Returns:
            None
        """

        features = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(self.paths), self.batch_size),
                desc="> Computing features...",
            ):
                outputs = self.model(
                    **self.feature_extractor(
                        [
                            Image.open(path).convert("RGB")
                            for path in self.paths[i : i + self.batch_size]
                        ],
                        return_tensors="pt",
                    ),
                    output_hidden_states=True,
                ).pooler_output
                features.append(outputs)
            # Concatenate features into single np.array
            self.features = np.concatenate(features, axis=0)

    def sequence_images(self):
        """
        Generates a sequence of images based on the Traveling Salesman Problem (TSP) route.
        This method computes the TSP route using the provided features and time limit for the TSP solver.
        The computed route is stored in the instance variable `self.route`.

        Steps:
        1. Prints a message indicating the start of TSP route computation.
        2. Computes the TSP route using the `tsp` function with the instance's features and time limit.
        3. Stores the computed route in the instance variable `self.route`.

        Note:
        - The computed route is stored as a NumPy array.
        - The route can be saved to a file by uncommenting the `np.save` line.

        Returns:
        None
        """

        # Obtain TSP route
        pbar = tqdm(total=1, desc="> Computing TSP route...", position=0, leave=True)
        self.route = np.array(tsp(self.features, time_limit=self.tsp_solver_limit))
        pbar.update(1)
        # np.save(f'route-{self.output_name}.npy', route)

    def animate(
        self,
        framerate=None,
    ):
        """
        Animates a sequence of images by creating a video and a GIF from them.

        Parameters:
        framerate (int, optional): The framerate for the animation. If not provided, the default framerate of the instance will be used.

        Returns:
        IPython.display.Image: An IPython Image object pointing to the generated GIF.

        Raises:
        OSError: If there is an error in creating directories or saving images.

        Notes:
        - This method requires the 'ffmpeg' and 'tqdm' libraries.
        - The method will create a temporary directory to store the sequential images
          and will delete it after the video and GIF are created.
        - The method will extract features and sequence images if they are not already present in the instance.
        """

        if "features" not in self.__dict__:
            self.extract_features()

        if "route" not in self.__dict__:
            self.sequence_images()

        # Save sequential images in new folder
        os.makedirs(f"./.tmp/{self.output_name}", exist_ok=True)
        for i, path in enumerate(
            tqdm(np.array(self.paths)[self.route], desc="> Saving frames...")
        ):
            img = Image.open(path).convert("RGB")

            # Delete temporary folder
            img.save(f"./.tmp/{self.output_name}/{i}.png")
            i += 1

        pbar = tqdm(total=1, desc="> Creating video...", position=0, leave=True)

        # Create parent directory of self.output_name if it doesn't exist
        if os.path.dirname(self.output_name) != "":
            os.makedirs(os.path.dirname(self.output_name), exist_ok=True)

        (
            ffmpeg.input(
                f"./.tmp/{self.output_name}/*.png",
                pattern_type="glob",
                framerate=framerate if framerate is not None else self.framerate,
            )
            .output(
                f"{self.output_name}.mp4",
                vcodec="libx264",
                vf=f"scale={self.resolution[0]}:{self.resolution[1]}:force_original_aspect_ratio=decrease:eval=frame,pad={self.resolution[0]}:{self.resolution[1]}:-1:-1:eval=frame",
            )
            .run(quiet=True, overwrite_output=True)
        )

        (
            ffmpeg.input(f"{self.output_name}.mp4")
            .output(
                f"{self.output_name}.gif",
                vf=f"fps={framerate if framerate is not None else self.framerate},scale={self.resolution[0]}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            )
            .run(quiet=True, overwrite_output=True)
        )

        pbar.update(1)

        shutil.rmtree(f".tmp/")

        return IPython.display.Image(url=f"{self.output_name}.gif")


def main():

    parser = argparse.ArgumentParser(description="Create TSP animation from images.")
    parser.add_argument(
        "input_folder", type=str, help="Path to the folder containing input images."
    )
    parser.add_argument(
        "output_name", type=str, help="Name of the output animation file."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images to process in a batch.",
    )
    parser.add_argument(
        "--tsp_solver_limit",
        type=int,
        default=120,
        help="Time limit for the TSP solver in seconds.",
    )
    parser.add_argument(
        "--resolution",
        type=tuple,
        default=(2000, 2000),
        help="Resolution of the output animation.",
    )
    parser.add_argument(
        "--framerate", type=int, default=8, help="Frame rate of the output animation."
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Maximum number of items to process.",
    )

    args = parser.parse_args()

    animation = TSPAnimation(
        input_folder=args.input_folder,
        output_name=args.output_name,
        batch_size=args.batch_size,
        tsp_solver_limit=args.tsp_solver_limit,
        resolution=args.resolution,
        framerate=args.framerate,
        max_items=args.max_items,
    )

    animation.animate()
