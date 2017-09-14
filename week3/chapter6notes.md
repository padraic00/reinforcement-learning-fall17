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
