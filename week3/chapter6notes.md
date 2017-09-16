Reinforcement Learning Notes
===================

Below are all the notes taken from Chapter 6 onward for Sutton and Barto *Reinforcement Learning*.

----------


## Chapter 6


* Temporal Difference Learning is the idea central and novel to reinforcement learning
* Combination of Monte Carlo and DP ideas
	* Can learn directly from experience like Monte Carlo
	* Update estimates in a bootstrapped methods like DP

### Section 6.1: TD Prediction

* While both TD and Monte Carlo follow some policy $ \pi $ to update their estimate their estimates $ v $ of $ v_\pi $ for the nonterminal states $ S_t $ 
* However Monte Carlo methods must wait until the return is known and then use that as a target for $ V(S_t) $ 
	* $ V(S_t) \leftarrow V(S_t) + \alpha[G_t - V(S_t)]$ with $G_t$ as the actual return following time $t$, and $\alpha$ is a constant step-size parameter. This is called $constant-\alpha\,MC$
* TD methods need only wait till the next time step to increment $V(S)$. The simplest TD Learning Method is called $TD(0)$
	* $V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1}+\gamma V(S_{t+1})-V(S_t)$
	* The effective target for the Monte Carlo update is $G_t$, for TD it is $R_{t+1}+\gamma V_t(S_{t+1})$

>**TD(0):**
![TD(0) Code](https://drive.google.com/uc?id=0B3oXhBTHfrfdYU9iT3B4T2pDc0U)

* Refer to TD updates as *sample backups* because they involve looking ahead to a sample successor state, use the successor value and reward along the way to compute a backed-up value, and adjust the original state accordingly
>**Example 6.1: Driving Home**<
> - When driving home you try to estimate how long you will take
> - Beginning states are defined by time of day, day of week
> - E.g. On Friday, you leave at 6 PM, and you estimate it'll take 30 minutes
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - When you reach your car at 6:05, it starts to rain, which slows traffic so you adjust the estimate to 35 minutes from then, or 6:40
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - 15 minutes later, you have made up time on the highway so you readjust to 6:35
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - On the secondary road, you gt stuck behind a truck all the way till you reach your street at 6:40
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - 3 minutes later you reach home
| State     | Elapsed Time | Predicted Time to Go   |Predicted Total|
| :---------------------------- | --: | :--: | --:  |
| leaving office, friday at 6	| 0  |  30   |	30	|
| reach car, raining| $12    	| 5  |	35	 |	40	|
| exiting highway     		 	| 20  |  15  |	35	|
| 2ndary road, behind truck	 	| 30  |	10	 |	40	|
| entering home street			| 40  | 3	 |	43	|
| arrive home					| 43  |	0	 |	43	|

> - Rewards=elapsed times on each leg
> - Value = *expected* time to go
> - Monte Carlo is effectively a plot of the Predicted Total column:
> ![Plot](https://drive.google.com/uc?id=0B3oXhBTHfrfdR1QtV3lWc080ak0)
> - Any changes to the policy must be made after coming home. Is this really necessary?
> - Suppose instead another day you estimate it will take 30 minutes, but you get stuck in a massive jam
> - 25 minutes after leaving the office you are still bumper-to-bumper on the highway, now your estimate is another 25 until you are home
> - While sitting around in traffic, you have made the decision that the 30 minutes initial is too optimistic. What is the point of waiting to adjust until you get home?
> - **TD** would point you to adjust every estimate toward the estimate following it
> - Below is a plot of what TD recommends as opposed to the MC above. Each error is proportional to the change over time of the prediction, or the *temporal differences* in predictions:
> ![TD Plot](https://drive.google.com/uc?id=0B3oXhBTHfrfdQUtIdElHVnhaSmM)
> - There are many computational reasons why it is advantageous to learn based on current predictions rather than waiting till termination

### Section 6.2: Advantages of TD Models
* TD methods learn a guess from a guess-they *bootstrap*
* TD methods are better than DP methods because they do not require a complete model of the environment
* TD methods are fully online as opposed to Monte Carlo's episodic task completion learning
	* With applications with long episodes, this can be significantly better
* TD methods do mathematically converge to right policy. If both MC and TD converge, which converges faster?
* Mathematically, this hasn't been proven by any means. However, in practice, TD methods are quicker than constant-$\alpha$ MC methods

>**TD(0):**
> - Consider the following random walk Markov process:
> ![Random Walk](https://drive.google.com/uc?id=0B3oXhBTHfrfdVmFyNEJ0ZTZ0bjg)
> - Lets compare the prediction capabilities of TD(0) and constant-$\alpha$ MC
> - All episodes start in state C, and go left or right by one with equal probability
> - Episodes terminate when they reach the terminal points on the extreme left or right, only the right results in reward +1
> - Thus, the expected value of all states A-D are, in order, $\frac{1}{6},\, \frac{2}{6},\, \frac{3}{6},\, \frac{4}{6},\, \frac{5}{6}$
> - Below is the learned values from TD(0) labeled by the number of runs:
> ![Figure 6.6](https://drive.google.com/uc?id=0B3oXhBTHfrfdQnktbmRNTTJIbUE)
> - Below is a comparison of the learning curves of TD(0) and constant-$\alpha$ MC in a RMS error plot:
> ![Figure 6.7](https://drive.google.com/uc?id=0B3oXhBTHfrfdX1puZVZoRFhSb0E)
> - **Seems obviously like a good application for random walk finance methods. Imagine that state C is the current price of an asset, and state E is a target increase, say a 10% return. One could simply watch the asset as it returns to C, and quantify the value of other states until it hits E. The learner will have essentially figured out the time horizon and patterns for that particular asset.**

### 6.3: Optimality of TD(0)
* Suppose that only a finite amount of experience in episodes or time steps is available
* Commonly, the data is run over and over convergence
	* In this case, the increments to the value function are stashed until that run of data is finished
	* Then, when that run is finished, the increments are summed and modify the value function
	* This is called *batched updating*
* Under this scheme, TD(0) converges to a single answer independent of the step-size $\alpha$, as long as its chosen to be sufficiently small
* MC also converges in batch updating, but to a less correct answer:
![Figure 6.8](https://drive.google.com/uc?id=0B3oXhBTHfrfdSDhJMEQxWFFDRTQ)
* Technically, constant-$\alpha$ MC methods do get error estimates from returns in inside the value policy, but the TD is optimal in a much better way to predict returns
>**Example 6.4:**
> - Imagine that you are the predictor of returns for an unknown Markov reward process
> - Observe the following eight episodes:
> ![Example 6.4](https://drive.google.com/uc?id=0B3oXhBTHfrfdTmFjTkVhbnMxR1E)
> - This translates to the first episode starting in state A, transitioning to B with reward 0, and then terminated from B with reward 0, the other episodes terminated immediately
> - Any logical predictor would say $V(B)=\frac{3}{4}$, because all episodes of B terminated immediately, 6 out of 8 episodes containing B terminated  at value 1 and the others at value 0
> - V(A) can only be estimated based on the fact that the only time it got state A, it went to state B, meaning that $V(A)=\frac{3}{4}$
> - The other option is that because the net reward at the end of the state with A is 0, $V(B)=0$. This is the answer batch Monte Carlo gives
> - While batch Monte Carlo gives the least minimum squared error, actually zero error, but we still think the first estimate makes more sense

* The difference is clear: batched Monte Carlo always find the estimates that minimize mean-squared error on the training set, while batched TD(0) will always find the estimates that would be exactly correct for the maximum-likelihood model of the Markov process
*  The *maximum-likelihood estimate* of a parameter is the parameter whose probability of generating the data is greatest
	* In this case, the MLE is the model of the Markov Process: the estimated transition probability from i to j is the fraction of observed transition from i that went to j, and the associated expected reward is the average of the rewards received from such a transition
* Given this model, the estimate of the value function that would be correct if the model was correct can be computed
	* This is called the *certainty-equivalence estimate*, because it is equivalent to assuming that the estimate of the underlying process was known with certainty rather than being approximated
* In general TD(0) converges to this
* This explains why TD methods converge more quickly than MC
	* In batch form, TD(0) computes the true certainty-equivalence estimate
* Although the certainty-equivalence estimate is optimal in some ways, it isn't feasible to compute it directly
	* The process takes $N^2$ memory and $N^3$ time steps
* TD(0) are suprisingly able to estimate it with memory N and just repeated computation over the set 

### 6.4: Sarsa- On-Policy TD
* Lets look at TD for policy control
* The first step is to compute an action-value function rather than a state-value function
	* Recall that an episode consists of an alternating sequence of states and state-action pairs
* Previously, we consider transitions from state to state and learned the values. Instead lets, consider transitions from state-action pair to pair, and learn the values of these pairs
* Theorems assuring converegence for TD(0) still apply for the following equation:
![Equation 6.5](https://drive.google.com/uc?id=0B3oXhBTHfrfdR01zdkNRekU1WkU)
- This rule uses every element of the quintuple: $(S_t,\, A_t,\, R_{t+1}\,  S_{t+1},\,A_{t+1})$
	- Gives rise to the name Sarsa
- As with all on-policy, we continually estimate $q_{\pi} $ for the behavior policy $\pi$, and at the same time modify $\pi$ towards $q_{\pi} $
![Figure 6.9](https://drive.google.com/uc?id=0B3oXhBTHfrfdRE9Dc3p2eWFSdzA)
### 6.5: Q-Learning- Off-Policy TD Control
* Q-Learning is the off policy TD Control method. One step Q Learning is the following equation:
$Q(S_t,A_t)\leftarrow Q(S_t,A_t) + \alpha [R_{t+1}+\gamma max_a(Q(S_{t+1},a))-Q(S_t,A_t)]$
* Q directly approximates $q_{*}$, the optimal value function
* Q will converge with probability 1 to $q_{*}$ if all steps continue to be updated
* The Q-Learning Algorithm is this:
![Figure 6.12](https://drive.google.com/uc?id=0B3oXhBTHfrfdNk5IdEwxcmE0U28)
* The Q-Learning backup diagram is this:
![Figure 6.14](https://drive.google.com/uc?id=0B3oXhBTHfrfdQUhVSFFUOTVRSUE)


