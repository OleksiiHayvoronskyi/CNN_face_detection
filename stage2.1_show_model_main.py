# Import functions: 'load_data()' and 'create_cnn_model()':
import stage2_create_model


# Load data
def main():
    # Load data
    data, target = stage2_create_model.load_data()

    # Create model
    model = stage2_create_model.create_cnn_model(data.shape[1:])

    # Visualize model
    stage2_create_model.visualize_model(model)


# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()
