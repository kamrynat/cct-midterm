import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load the plant knowledge data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing plant knowledge data
        
    Returns:
    --------
    numpy.ndarray
        A binary matrix of shape (n_informants, n_questions) containing responses
    list
        List of informant IDs
    list
        List of question IDs
    """
    # Read the data
    df = pd.read_csv(file_path)
    
    print("CSV file columns:", df.columns.tolist())
    
    # The first column is the informant ID
    informant_column = "Informant"
    
    # Extract informant IDs
    informant_ids = df[informant_column].unique()
    
    # All other columns (PQ1-PQ20) are the questions
    question_columns = [col for col in df.columns if col != informant_column]
    
    # Set the informant column as the index
    data_df = df.set_index(informant_column)
    
    # Get only the question columns
    data_df = data_df[question_columns]
    
    # Get question IDs
    question_ids = data_df.columns.tolist()
    
    # Convert to numpy array
    data_matrix = data_df.values
    
    # Make sure responses are binary
    if not np.isin(data_matrix, [0, 1]).all():
        print("Warning: Not all responses are binary (0 or 1). Converting non-binary values.")
        data_matrix = (data_matrix > 0).astype(int)
    
    print(f"\nExtracted data matrix of shape {data_matrix.shape} ({len(informant_ids)} informants × {len(question_ids)} questions)")
    
    return data_matrix, informant_ids.tolist(), question_ids


def run_cct_model(X, n_chains=4, n_samples=2000):
    """
    Implement the CCT model using PyMC and perform MCMC sampling.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Binary response matrix of shape (n_informants, n_questions)
    n_chains : int
        Number of MCMC chains
    n_samples : int
        Number of samples per chain
    
    Returns:
    --------
    arviz.InferenceData
        Trace of the MCMC sampling
    """
    # Get dimensions
    N, M = X.shape  # N informants, M questions
    
    # Create PyMC model
    with pm.Model() as cct_model:
        # Define priors for competence (D)
        # Using Beta(2,2) as a reasonable prior - it favors values around 0.5 
        # but allows for the full range between 0 and 1
        D = pm.Beta("D", alpha=2, beta=2, shape=N)
        
        # Define priors for consensus answers (Z)
        # Using Bernoulli(0.5) for Z - no prior belief about the correct answer
        Z = pm.Bernoulli("Z", p=0.5, shape=M)
        
        # Reshape D for broadcasting with Z
        D_reshaped = D[:, None]
        
        # Calculate response probabilities
        # p_ij = Z_j * D_i + (1 - Z_j) * (1 - D_i)
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)
        
        # Define the likelihood using the calculated probabilities
        X_obs = pm.Bernoulli("X_obs", p=p, observed=X)
        
        # Sample from the posterior
        trace = pm.sample(
            n_samples, 
            chains=n_chains,
            tune=1000,
            return_inferencedata=True,
            target_accept=0.9
        )
    
    return trace


def analyze_results(trace, informant_ids, question_ids):
    """
    Analyze the MCMC sampling results.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        Trace from MCMC sampling
    informant_ids : list
        List of informant IDs
    question_ids : list
        List of question IDs
    
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # Check convergence
    summary = az.summary(trace)
    
    # Extract posterior samples
    posterior_samples = trace.posterior
    
    # Get competence estimates (D)
    D_samples = posterior_samples["D"]
    D_mean = D_samples.mean(dim=["chain", "draw"]).values
    
    # Get consensus answer estimates (Z)
    Z_samples = posterior_samples["Z"]
    Z_mean = Z_samples.mean(dim=["chain", "draw"]).values
    Z_mode = (Z_mean > 0.5).astype(int)  # Most likely answer
    
    # Create results dictionary
    results = {
        "competence": {informant_ids[i]: D_mean[i] for i in range(len(informant_ids))},
        "consensus_prob": {question_ids[i]: Z_mean[i] for i in range(len(question_ids))},
        "consensus_answer": {question_ids[i]: Z_mode[i] for i in range(len(question_ids))},
        "r_hat": summary["r_hat"].max()
    }
    
    # Find most and least competent informants
    most_competent = informant_ids[np.argmax(D_mean)]
    least_competent = informant_ids[np.argmin(D_mean)]
    
    results["most_competent"] = (most_competent, results["competence"][most_competent])
    results["least_competent"] = (least_competent, results["competence"][least_competent])
    
    return results


def calculate_majority_vote(X, question_ids):
    """
    Calculate the simple majority vote for each question.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Binary response matrix of shape (n_informants, n_questions)
    question_ids : list
        List of question IDs
    
    Returns:
    --------
    dict
        Dictionary mapping question IDs to majority vote answers
    """
    # Calculate the mean response for each question
    maj_vote_prob = X.mean(axis=0)
    
    # Determine majority vote answer (0 or 1)
    maj_vote = (maj_vote_prob > 0.5).astype(int)
    
    return {question_ids[i]: maj_vote[i] for i in range(len(question_ids))}


def plot_competence_distribution(trace, informant_ids):
    """
    Visualize the posterior distribution of competence for each informant.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        Trace from MCMC sampling
    informant_ids : list
        List of informant IDs
    """
    plt.figure(figsize=(12, 8))
    az.plot_posterior(trace, var_names=["D"], hdi_prob=0.95)
    plt.title("Posterior Distributions for Informant Competence")
    plt.tight_layout()
    plt.savefig("competence_posterior.png")
    plt.close()
    
    # Plot ordered by mean competence
    posterior_samples = trace.posterior
    D_mean = posterior_samples["D"].mean(dim=["chain", "draw"]).values
    
    # Sort by mean competence
    sort_idx = np.argsort(D_mean)[::-1]
    sorted_ids = [informant_ids[i] for i in sort_idx]
    sorted_means = D_mean[sort_idx]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(sorted_ids)), sorted_means)
    plt.xticks(range(len(sorted_ids)), sorted_ids, rotation=90)
    plt.xlabel("Informant ID")
    plt.ylabel("Mean Competence (D)")
    plt.title("Informants Ranked by Competence")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("competence_ranking.png")
    plt.close()


def plot_consensus_distribution(trace, question_ids):
    """
    Visualize the posterior distribution of consensus answers.
    
    Parameters:
    -----------
    trace : arviz.InferenceData
        Trace from MCMC sampling
    question_ids : list
        List of question IDs
    """
    plt.figure(figsize=(12, 8))
    az.plot_posterior(trace, var_names=["Z"], hdi_prob=0.95)
    plt.title("Posterior Distributions for Consensus Answers")
    plt.tight_layout()
    plt.savefig("consensus_posterior.png")
    plt.close()
    
    # Plot ordered by certainty
    posterior_samples = trace.posterior
    Z_mean = posterior_samples["Z"].mean(dim=["chain", "draw"]).values
    
    # Calculate certainty (distance from 0.5)
    certainty = np.abs(Z_mean - 0.5) + 0.5
    
    # Sort by certainty
    sort_idx = np.argsort(certainty)[::-1]
    sorted_ids = [question_ids[i] for i in sort_idx]
    sorted_means = Z_mean[sort_idx]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(sorted_ids)), sorted_means)
    plt.xticks(range(len(sorted_ids)), sorted_ids, rotation=90)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.xlabel("Question ID")
    plt.ylabel("Consensus Answer Probability (Z)")
    plt.title("Questions Ranked by Consensus Answer Certainty")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("consensus_certainty.png")
    plt.close()


def compare_with_majority(cct_results, majority_results):
    """
    Compare CCT consensus answers with majority vote.
    
    Parameters:
    -----------
    cct_results : dict
        Dictionary containing CCT consensus answers
    majority_results : dict
        Dictionary containing majority vote answers
    
    Returns:
    --------
    float
        Agreement rate between CCT and majority vote
    list
        List of questions where CCT and majority vote disagree
    """
    # Get CCT consensus answers
    cct_answers = cct_results["consensus_answer"]
    
    # Count agreements
    agreements = sum(cct_answers[q] == majority_results[q] for q in cct_answers)
    
    # Calculate agreement rate
    agreement_rate = agreements / len(cct_answers)
    
    # Find disagreements
    disagreements = [q for q in cct_answers if cct_answers[q] != majority_results[q]]
    
    return agreement_rate, disagreements


def main():
    """
    Main function to run the CCT analysis.
    """
    # Load the data
    data_file = "../data/plant_knowledge.csv"
    try:
        X, informant_ids, question_ids = load_data(data_file)
        print(f"Loaded data with {len(informant_ids)} informants and {len(question_ids)} questions")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nAttempting to show the CSV file contents for debugging:")
        try:
            with open(data_file, 'r') as f:
                print("First 10 lines of the CSV file:")
                for i, line in enumerate(f):
                    if i < 10:
                        print(line.strip())
                    else:
                        break
            return
        except Exception as e2:
            print(f"Could not open the CSV file: {e2}")
            return
    
    # Run the CCT model
    print("Running CCT model with PyMC (this may take a few minutes)...")
    trace = run_cct_model(X)
    
    # Analyze results
    print("Analyzing results...")
    results = analyze_results(trace, informant_ids, question_ids)
    
    # Calculate majority vote
    majority_results = calculate_majority_vote(X, question_ids)
    
    # Compare CCT with majority vote
    agreement_rate, disagreements = compare_with_majority(results, majority_results)
    
    # Print results
    print("\n===== MODEL RESULTS =====")
    print(f"Model convergence (max r-hat): {results['r_hat']:.4f}")
    print(f"{'Model converged' if results['r_hat'] < 1.1 else 'Model did NOT converge properly'}")
    
    print("\n----- Informant Competence -----")
    for informant, competence in sorted(results["competence"].items(), key=lambda x: x[1], reverse=True):
        print(f"Informant {informant}: {competence:.4f}")
    
    print(f"\nMost competent informant: {results['most_competent'][0]} ({results['most_competent'][1]:.4f})")
    print(f"Least competent informant: {results['least_competent'][0]} ({results['least_competent'][1]:.4f})")
    
    print("\n----- Consensus Answers -----")
    for question, prob in sorted(results["consensus_prob"].items(), key=lambda x: abs(x[1]-0.5), reverse=True):
        answer = results["consensus_answer"][question]
        print(f"Question {question}: {answer} (probability: {prob:.4f})")
    
    print("\n----- Comparison with Majority Vote -----")
    print(f"Agreement rate: {agreement_rate:.2%}")
    if disagreements:
        print(f"Disagreements on {len(disagreements)} questions: {disagreements}")
    else:
        print("No disagreements between CCT and majority vote")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_competence_distribution(trace, informant_ids)
    plot_consensus_distribution(trace, question_ids)
    
    print("\nAnalysis complete! Check the output directory for plots.")


if __name__ == "__main__":
    main()