'''
b_labels is concatenation of observation and goals
a gaussian noise is added in the b_actions before passing into the train_step

nf_model = RealNVP()
encoder = RealNVPEncoder()
prior = create_prior(shape -> action_dim)
actor_state is the TrainState combining the parameters and optimizer

FUNCTION train_step(state -> actor_state,
                    b_actions -> noisy actions
                    b_labels -> concatenated vector of observation and goals)

    FUNCTION loss_fn(params):
        z, log_dets = nf_model(x = b_actions,
                                y = encoder(b_labels))

        loss = -(prior.log_prob(z) + log_dets).mean() -- Eq 5 in the paper
    
    (loss, z, logdets) = loss_fn()
    return new_state, loss, z, logdets

        ## Explanation - 

            In the paper, they learn the conditional distribution p(x|y) thats why 
            nf_model takes in y as an additional input. 

FUNCTION get_action(state -> actor_state
                    observation, 
                    goal
                    sample_key)
    
    prior_sample = prior.sample() -> samples from gaussian of dimension (N, action_dim)
    obs_goal_z = encoder([obs, goal])
    action, _ = nf_model(x -> prior_sample,
                        y -> obs_goal_z,
                        reverse = True)

FUNCTION get_denoised_action(state -> actor_state
                    observation, 
                    goal
                    sample_key)

    def log_prob_fn(x, y):
        z, logdets = nf_model.apply(state.params['model'], x=x, y=y)
        logprob = prior.log_prob(z) + logdets
        return logprob.sum()

    
    prior_sample = prior.sample() -> samples from gaussian of dimension (N, action_dim)
    obs_goal_z = encoder([obs, goal])
    action, _ = nf_model(x -> prior_sample,
                        y -> obs_goal_z,
                        reverse = True)

    action = stop_gradient(action)
    action_score = jax.grad(log_prob_fn)(action, observation_goal_z)
    action = action + args.noise_std**2 * action_score

    ## explanation 

    so the NF models have the following flow - 

        z = p(a|y)
        a = p-1(z|y)
        
'''