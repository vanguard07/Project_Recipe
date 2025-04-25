import React, { useState } from 'react';
import axios from 'axios';
import './RecipeStore.css';

const RecipeStore = () => {
  const [url, setUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;

    setIsLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await axios.post('http://localhost:8000/recipe/store', { prompt: url });
      setResult(response.data);
    } catch (err) {
      console.error('Error storing recipe:', err);
      setError(err.response?.data?.detail || 'An error occurred while storing the recipe');
    } finally {
      setIsLoading(false);
    }
  };

  const renderRecipeDetails = (recipe) => {
    return (
      <div className="recipe-details">
        <h3>{recipe.title}</h3>
        
        {recipe.cuisine && <p><strong>Cuisine:</strong> {recipe.cuisine}</p>}
        {recipe.meal_type && <p><strong>Meal Type:</strong> {recipe.meal_type}</p>}
        
        <div className="recipe-time">
          {recipe.prep_time && <span><strong>Prep:</strong> {recipe.prep_time}</span>}
          {recipe.cook_time && <span><strong>Cook:</strong> {recipe.cook_time}</span>}
        </div>
        
        <div className="recipe-section">
          <h4>Ingredients</h4>
          <ul>
            {recipe.ingredients.map((ingredient, index) => (
              <li key={index}>{ingredient}</li>
            ))}
          </ul>
        </div>
        
        <div className="recipe-section">
          <h4>Instructions</h4>
          <ol>
            {recipe.instructions.map((instruction, index) => (
              <li key={index}>{instruction}</li>
            ))}
          </ol>
        </div>
        
        {recipe.tags && recipe.tags.length > 0 && (
          <div className="recipe-tags">
            {recipe.tags.map((tag, index) => (
              <span key={index} className="recipe-tag">{tag}</span>
            ))}
          </div>
        )}
        
        {recipe.source_url && (
          <p className="recipe-source">
            <a href={recipe.source_url} target="_blank" rel="noopener noreferrer">
              Original Source
            </a>
          </p>
        )}
      </div>
    );
  };

  return (
    <div className="recipe-store">
      <div className="store-header">
        <h2>Store New Recipe</h2>
        <p>Enter a URL of a recipe to store it in the database</p>
      </div>
      
      <form className="store-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Paste recipe URL here..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !url.trim()}>
          {isLoading ? 'Processing...' : 'Store Recipe'}
        </button>
      </form>
      
      <div className="store-result">
        {isLoading && (
          <div className="store-loading">
            <div className="loading-spinner"></div>
            <p>Analyzing recipe page...</p>
          </div>
        )}
        
        {error && (
          <div className="store-error">
            <h3>Error</h3>
            <p>{error}</p>
          </div>
        )}
        
        {result && (
          <div className="store-success">
            <h3>Recipe Stored Successfully!</h3>
            {renderRecipeDetails(result)}
          </div>
        )}
      </div>
    </div>
  );
};

export default RecipeStore;