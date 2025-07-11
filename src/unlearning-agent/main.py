import os
import argparse
import logging
import time
import json
from pathlib import Path
from dotenv import load_dotenv

from .agent import UnlearningAgent

load_dotenv()

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool):
    """Configures the root logger for the application."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)
    if not verbose:
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)


def load_unlearning_list(file_path: Path) -> list[str]:
    """
    Loads a list of unlearning subjects from a file.
    Supports JSON lists or plain text files with one subject per line.
    """
    logger.info(f"Loading unlearning subjects from: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"The specified unlearning file does not exist: {file_path}")

    content = file_path.read_text(encoding='utf-8')
    
    # Try parsing as JSON first
    try:
        data = json.loads(content)
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            logger.debug("Successfully parsed file as a JSON list of strings.")
            return data
        else:
            raise ValueError("JSON file is not a flat list of strings.")
    except json.JSONDecodeError:
        logger.debug("File is not valid JSON, falling back to line-by-line text parsing.")
        return [line.strip() for line in content.splitlines() if line.strip()]
    except ValueError as e:
        logger.error(f"Invalid JSON format in '{file_path}': {e}")
        raise


def main():
    """Main function to run the Unlearning Agent from the command line."""
    
    parser = argparse.ArgumentParser(
        description="A CLI for the Unlearning Agent to process a query while removing specified subjects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    core_group = parser.add_argument_group('Core Arguments')
    core_group.add_argument("prompt", type=str, help="The user query to process.")
    core_group.add_argument(
        "--unlearning-file",
        type=Path, 
        required=True,
        help="Path to a JSON or TXT file containing the list of subjects to unlearn."
    )

    model_group = parser.add_argument_group('Model and API Configuration')
    model_group.add_argument(
        "--hf-check-model",
        type=str,
        default=None,
        metavar="MODEL_NAME_OR_PATH",
        help="Optional Hugging Face model to use for the initial subject check."
    )
    model_group.add_argument(
        "--prompt-dir", 
        type=Path, 
        default=Path("prompts"), 
        help="Directory containing prompt templates."
    )
    model_group.add_argument(
        "--openai-api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API Key. Can also be set via OPENAI_API_KEY environment variable."
    )
    model_group.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face Token. Can also be set via HF_TOKEN environment variable."
    )

    control_group = parser.add_argument_group('Control and Debugging')
    control_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose DEBUG level logging for detailed output."
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    
    if not args.openai_api_key:
        logger.critical("FATAL: OpenAI API key not found. Please provide it via the --openai-api-key argument or set the OPENAI_API_KEY environment variable.")
        return 1

    if args.hf_check_model and not args.hf_token:
        logger.warning("A Hugging Face check model is specified, but no HF token is provided. This may fail for gated or private models.")

    try:
        unlearning_list = load_unlearning_list(args.unlearning_file)
        logger.info(f"Successfully loaded {len(unlearning_list)} subjects to unlearn.")

        
        start_time = time.time()
        
        agent = UnlearningAgent(
            unlearning_subjects=unlearning_list,
            hf_check_model_name_or_path=args.hf_check_model,
            openai_api_key=args.openai_api_key,
            hf_token=args.hf_token,
            prompt_dir=str(args.prompt_dir) 
        )
        
        final_response = agent.invoke(args.prompt)
        
    
        print("\n" + "="*20 + " Final Agent Response " + "="*20)
        print(final_response)
        print("="*62)

    except FileNotFoundError as e:
        logger.error(f"A required file was not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"A value error occurred, likely due to malformed input file: {e}")
        return 1
    except Exception as e:
        # A catch-all for unexpected errors from the agent or libraries
        logger.critical(f"An unhandled error occurred during agent execution: {e}", exc_info=args.verbose)
        return 1
    
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds.")
    return 0 

if __name__ == '__main__':
    import sys
    sys.exit(main())
