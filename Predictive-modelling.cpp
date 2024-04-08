#include<iostream>
#include<vector>
using namespace std;
// Define a struct to represent a data point with features and target
struct DataPoint {
    vector<double> features;
    double target;
};

// Function to perform linear regression
void linearRegression(const vector<DataPoint>& data, vector<double>& coefficients, double& intercept) {
    int numFeatures = data[0].features.size();
    int numSamples = data.size();

    // Initialize the coefficients vector with zeros
    coefficients.assign(numFeatures, 0.0);
    intercept = 0.0;

    // Calculate means for normalization
    vector<double> featureMeans(numFeatures, 0.0);
    double targetMean = 0.0;
    for (const auto& point : data) {
        targetMean += point.target;
        for (int i = 0; i < numFeatures; ++i) {
            featureMeans[i] += point.features[i];
        }
    }
    targetMean /= numSamples;
    for (int i = 0; i < numFeatures; ++i) {
        featureMeans[i] /= numSamples;
    }

    // Perform gradient descent
    const double learningRate = 0.01;
    const int maxIterations = 1000;
    for (int iter = 0; iter < maxIterations; ++iter) {
        vector<double> deltaCoefficients(numFeatures, 0.0);
        double deltaIntercept = 0.0;

        // Compute predictions and errors
        for (const auto& point : data) {
            double prediction = intercept;
            for (int i = 0; i < numFeatures; ++i) {
                prediction += coefficients[i] * (point.features[i] - featureMeans[i]);
            }
            double error = prediction - (point.target - targetMean);

            // Update deltas
            deltaIntercept += error;
            for (int i = 0; i < numFeatures; ++i) {
                deltaCoefficients[i] += error * (point.features[i] - featureMeans[i]);
            }
        }

        // Update coefficients and intercept
        intercept -= learningRate * deltaIntercept / numSamples;
        for (int i = 0; i < numFeatures; ++i) {
            coefficients[i] -= learningRate * deltaCoefficients[i] / numSamples;
        }
    }
}

int main() {
    // Example dataset
    vector<DataPoint> data = {
        {{2.0, 3.0}, 5.0},
        {{3.0, 4.0}, 7.0},
        {{4.0, 5.0}, 9.0}
    };

    // Coefficients and intercept for the linear regression model
    vector<double> coefficients;
    double intercept;

    // Perform linear regression
    linearRegression(data, coefficients, intercept);

    // Print coefficients and intercept
    cout << "Coefficients: ";
    for (double coef : coefficients) {
        std::cout << coef << " ";
    }
    cout << endl;
    cout << "Intercept: " << intercept << endl;

    return 0;
}
