"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { BookOpen, LinkIcon, Plus, Loader2 } from "lucide-react"

type Recipe = {
  id: string
  title: string
  url: string
  description: string
  ingredients: string[]
  instructions?: string[]
  imageUrl?: string
  addedAt: Date
  nutritionInfo?: {
    calories?: number
    protein?: number
    fat?: number
  }
  cookingInfo?: {
    prepTime?: number
    cookTime?: number
    totalTime?: number
  }
  cuisine?: string
  mealType?: string
  tags?: string[]
}

export default function RecipesPage() {
  const [url, setUrl] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [recipes, setRecipes] = useState<Recipe[]>([])

  // Fetch recipes from backend when component mounts
  useEffect(() => {
  }, [])

  const extractRecipe = async (url: string) => {
    setIsLoading(true)

    try {
      // Call the backend /store endpoint
      const response = await fetch('http://localhost:8000/recipe/store', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      // Parse the returned recipe data
      const recipeData = await response.json();
      
      // Format the received recipe data to match our Recipe type
      const newRecipe: Recipe = {
        id: recipeData._id || Date.now().toString(),
        title: recipeData.title || 'Untitled Recipe',
        url: recipeData.source_url || url,
        description: `${recipeData.cuisine || ''} ${recipeData.meal_type || ''} recipe`.trim(),
        ingredients: recipeData.ingredients || [],
        instructions: recipeData.instructions || [],
        imageUrl: recipeData.image_url || "/placeholder.svg?height=200&width=300",
        addedAt: new Date(),
        nutritionInfo: {
          calories: recipeData.estimated_calories,
          protein: recipeData.protein_grams,
          fat: recipeData.fat_grams
        },
        cookingInfo: {
          prepTime: recipeData.prep_time_in_mins,
          cookTime: recipeData.cook_time_in_mins,
          totalTime: recipeData.total_time_in_mins
        },
        cuisine: recipeData.cuisine,
        mealType: recipeData.meal_type,
        tags: recipeData.tags
      }

      // Add the new recipe to our state
      setRecipes([newRecipe, ...recipes])
      setUrl("")
    } catch (error) {
      console.error("Error extracting recipe:", error)
      alert("Failed to extract recipe. Please check the URL and try again.")
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!url.trim() || isLoading) return

    try {
      // Basic URL validation
      new URL(url)
      extractRecipe(url)
    } catch (error) {
      alert("Please enter a valid URL")
    }
  }

  return (
    <div className="container py-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold mb-2">Recipe Storage</h1>
        <p className="text-muted-foreground">Add recipes by pasting a URL below</p>
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2 mb-8">
        <Input
          type="text"
          placeholder="Paste recipe URL here..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="flex-1"
          disabled={isLoading}
        />
        <Button type="submit" disabled={isLoading || !url.trim()}>
          {isLoading ? 
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Extracting...
            </> : 
            "Extract Recipe"
          }
        </Button>
      </form>

      {recipes.length > 0 ? (
        <div className="space-y-8">
          {recipes.map((recipe) => (
            <div key={recipe.id} className="border rounded-lg overflow-hidden bg-background">
              <div className="flex flex-col md:flex-row">
                <div className="flex-1 p-6">
                  <div className="mb-4">
                    <h2 className="text-2xl font-bold mb-2">{recipe.title.toUpperCase()}</h2>
                    <div className="flex items-center text-sm text-muted-foreground mb-2">
                      <LinkIcon className="h-3 w-3 mr-1" />
                      <a
                        href={recipe.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="truncate hover:underline"
                      >
                        {recipe.url}
                      </a>
                    </div>
                    <p className="text-muted-foreground">{recipe.description}</p>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {recipe.ingredients && recipe.ingredients.length > 0 && (
                      <div>
                        <h3 className="text-lg font-semibold mb-2">Ingredients</h3>
                        <ul className="list-disc pl-5 space-y-1">
                          {recipe.ingredients.map((ingredient, index) => (
                            <li key={index} className="text-sm">{ingredient}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {recipe.instructions && recipe.instructions.length > 0 && (
                      <div>
                        <h3 className="text-lg font-semibold mb-2">Instructions</h3>
                        <ol className="list-decimal pl-5 space-y-1">
                          {recipe.instructions.map((instruction, index) => (
                            <li key={index} className="text-sm">{instruction}</li>
                          ))}
                        </ol>
                      </div>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                    {recipe.nutritionInfo && (
                      <div className="border rounded p-4">
                        <h3 className="font-semibold mb-2">Nutrition Facts</h3>
                        <div className="space-y-1">
                          {recipe.nutritionInfo.calories && (
                            <div className="text-sm flex justify-between">
                              <span>Calories:</span> 
                              <span className="font-medium">{recipe.nutritionInfo.calories}</span>
                            </div>
                          )}
                          {recipe.nutritionInfo.protein && (
                            <div className="text-sm flex justify-between">
                              <span>Protein:</span> 
                              <span className="font-medium">{recipe.nutritionInfo.protein}g</span>
                            </div>
                          )}
                          {recipe.nutritionInfo.fat && (
                            <div className="text-sm flex justify-between">
                              <span>Fat:</span> 
                              <span className="font-medium">{recipe.nutritionInfo.fat}g</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {recipe.cookingInfo && (
                      <div className="border rounded p-4">
                        <h3 className="font-semibold mb-2">Cooking Time</h3>
                        <div className="space-y-1">
                          {recipe.cookingInfo.prepTime && (
                            <div className="text-sm flex justify-between">
                              <span>Prep Time:</span> 
                              <span className="font-medium">{recipe.cookingInfo.prepTime} mins</span>
                            </div>
                          )}
                          {recipe.cookingInfo.cookTime && (
                            <div className="text-sm flex justify-between">
                              <span>Cook Time:</span> 
                              <span className="font-medium">{recipe.cookingInfo.cookTime} mins</span>
                            </div>
                          )}
                          {recipe.cookingInfo.totalTime && (
                            <div className="text-sm flex justify-between">
                              <span>Total Time:</span> 
                              <span className="font-medium">{recipe.cookingInfo.totalTime} mins</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {recipe.tags && recipe.tags.length > 0 && (
                      <div className="border rounded p-4">
                        <h3 className="font-semibold mb-2">Tags</h3>
                        <div className="flex flex-wrap gap-1">
                          {recipe.tags.map((tag, index) => (
                            <span key={index} className="px-2 py-1 bg-muted text-xs rounded-full">
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div className="mt-6 flex justify-between items-center">
                    <span className="text-xs text-muted-foreground">Added {recipe.addedAt.toLocaleDateString()}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12 border rounded-lg bg-muted/20">
          <BookOpen className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <h2 className="text-xl font-semibold mb-2">No recipes yet</h2>
          <p className="text-muted-foreground mb-4">Add your first recipe by pasting a URL above</p>
          <Button variant="outline" onClick={() => setUrl("https://example.com/recipe")} className="mx-auto">
            <Plus className="mr-2 h-4 w-4" />
            Try with example URL
          </Button>
        </div>
      )}
    </div>
  )
}
