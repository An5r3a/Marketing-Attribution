# Channel Attribution needs Attention!
Marketing attribution is a way of measuring the value of the campaigns and channels that are reaching your potential customers. The point in time when a customer interacts with a channel is called a touchpoint, and a collection of touch points forms a user journey. Users journeys can be long with many touch points that makes it difficult to understand the true high and low impact of each interaction, which can result in an inaccurate division of credit and a false representation of marketing performance.
To overcome this problem, we introduce an attention mechanism for marketing attribution to relate different positions of a sequence of touch points to find out who they should pay more attention to.

## Limitation of traditional models

All attribution models have their pros and cons. Usually, we have to decide up front how we want to credit each touchpoint that resulted in a conversion. For example, linear model credits an equal share to the value between all touch points, time-decay model credits a decreasing percentage of value the further away in time a touchpoint is from the conversion, positional model credits 40% to the first and last touch points and the remaining 20% is evenly distributed to the touch points in between.
Luckily, we propose a deep learning model that is able to understand the interaction among all touch points in a user journey and calculates specific credits to each touchpoint depending on the impact each interaction has. The results of this model provide marketers with deeper insights into the importance of campaigns and channels, driving better marketing efficiency.

This article does not aim to explain the different numerical representations and mathematical operations in the self-attention models. The main content of this post is to walk you through the application of the self-attention model in marketing attribution and to provide a general overview of self-attention models.

## What is attention mechanism?
Attention mechanism is motivated by how we pay visual attention to different regions of an image or correlate words in one sentence. Human visual attention allows us to focus on a certain region with “high resolution” while perceiving the surrounding image in “low resolution” and then adjust the focal point or do the inference accordingly. Similarly, we can explain the relationship between words in one sentence or close context. Attention in deep learning can be interpreted as a vector of importance weights.

## How does Attention work?
Self-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. It was born to help memorise long source sentences in neural machine translation. For example, in order to predict or infer one element, such as a pixel in an image or a word in a sentence, we estimate using the attention vector how strongly it is correlated with other elements and take the sum of their values weighted by the attention vector as the approximation of the target.

With this in mind, we use attention mechanisms to we calculate a vector representation that focuses on specific components of the sequence of touch points for each user journey and the outputs of these interactions will allow us to assign more or less credit to a specific touchpoint in the journey.

## Comparing different attribution models

Let’s walk through an example. Say we have 50 different user journeys with 20 touch points over the period of time. We calculate the credits assigned to each touchpoint using a linear model, position-based model, time decay model and self-attention model.
We create a heatmap to visualise the credits related to different positions of a sequence of touch points. The x-axis indicates the 20 touch points for each of the 50 users. We can see that with the self-attention model as the touch points get more into the past, they get less important, while with the other models the distribution of the credits is more discrete. This indicates that self-attention models can actually identify which touch points are important in a user journey.

![image](https://user-images.githubusercontent.com/35504627/133578847-d45f95e3-5f35-4af2-9e63-7473b4d8918d.png)


Attribution credits heatmap comparison. The x-axis indicates the 20 touch points and y-axis represents the 50 users journey
Now, let’s see a more specific example for a user with 12 touch points.

![image](https://user-images.githubusercontent.com/35504627/133578887-46455982-32b1-4790-b081-05be48558e95.png)


## Conclusion

The self-attention model is able to find a better representative trend of interactions between the 12 touch points. Self-attention mechanism is a successful method that helps us make our attribution model interpretable and explain why it does what it does.
By leveraging a self-attention attribution model, we have eliminated the biases associated with traditional attribution mechanisms. We are able to divide credits by its relative importance in the user journey and have an accurate representation of marketing performance. Now that we have solid and trusted data behind attribution, we’re confident in using the results to inform and drive our marketing mix strategy and investment decisions.
