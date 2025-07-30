classdef UnscentedMPPIController < handle

    properties
        lambda                  % Temperature
        horizon                 % Prediction time
        n_samples               % Number of rollouts
        cov                     % Covariance
        R                       % Control weight matrix
        nu                      % Exploration variance

        model                   % Dynamics
        dt                      % Period
        
        obstacles

        control_sequence
        optimized_control_sequence

        goal
        state                   % Current state mean (x̄k)
        n_states = 5;           % Dimension of system's state [x, y, phi, v, steer]
        n_sigma_states = 3;      % Dimension of sigma points [x, y, phi]

        rollouts_states         % Saving trajectory
        rollouts_costs          % Saving cost
        rollouts_plot_handle    % Saving handle for visualization

        % Unscented Transform parameters
        alpha = 1;              % Sigma point spread parameter
        beta = 2;               % Parameter for prior knowledge of distribution
        kappa = 0;              % Scaling parameter

        % State estimation
        P                       % State covariance matrix (Σk)

        % Constraints
        max_vel = 5;
        max_steer = 0.5;

        % Robot physical parameter
        robot_radius = 0.1;
    end

    methods
        function self = UnscentedMPPIController(lambda, cov, nu, R, horizon, n_samples, model, dt, goal, obstacles)
            self.lambda = lambda;
            self.cov = cov;
            self.nu = nu;
            self.R = R;
            self.horizon = horizon;
            self.n_samples = n_samples;
            self.model = model;
            self.dt = dt;
            self.goal = goal;
            self.obstacles = obstacles;

            self.control_sequence = zeros(2, self.horizon);

            % Initialize Unscented Transform parameter
            self.initializeUnscentedTransform();
        end

        function initializeUnscentedTransform(self)
            % Initialize state covariance (Σ0 = 0.001I)
            self.P = eye(self.n_sigma_states) * 0.001;

            % Process noise covariance
            % self.Q = diag([100, 100, 50, 0, 0]);
        end

        function [sigma_points, weights_m, weights_c] = generateSigmaPoints(self, x, P)
            % Generate sigma points for Unscented Transform (6)
            n = self.n_sigma_states;
            lambda_ut = self.alpha^2 * (n + self.kappa) - n;

            % Calculate matrix square root
            try
                S = chol((n + lambda_ut)*P, 'lower');
            catch
                % If Cholesky fails, use eigenvalue decomposition
                [V, D] = eig((n + lambda_ut) * P);
                S = V * sqrt(max(D, 0)); % Ensure non-negative eigenvalues
            end

            % Generate sigma points (6)
            sigma_points = zeros(n, 2*n+1);
            sigma_points(:,1) = x;

            % S size is n*n, so subsituting each i column
            for i = 1:n
                sigma_points(:,i+1) = x + S(:,i);
                sigma_points(:,i+1+n) = x - S(:,i);
            end

            % Calculate weights (7)
            weights_m = zeros(1, 2*n+1);
            weights_c = zeros(1, 2*n+1);

            weights_m(1) = lambda_ut / (n + lambda_ut);
            weights_c(1) = weights_m(1) + (1 - self.alpha^2 + self.beta);

            for i = 2:2*n+1
                weights_m(i) = 1 / (2 * (n + lambda_ut));
                weights_c(i) = 1 / (2 * (n + lambda_ut));
            end
        end

        % Unscented Transform prediction step (8)
        function [x_pred, P_pred] = unscentedPredict(self, x, P, u)
            [sigma_points, weights_m, weights_c] = self.generateSigmaPoints(x, P);

            % Transform sigma points through dynamics
            n_sigma = size(sigma_points, 2);
            sigma_pred = zeros(self.n_states, n_sigma);     % ScriptX_(i+1)

            % Pass sigma point as row vector to model.step
            for i = 1:n_sigma
                sigma_pred(:,i) = self.model.step(u, self.dt, sigma_points(:,i)')';
            end

            % Calculate predicted mean
            x_pred = zeros(self.n_states, 1);
            for i = 1:n_sigma
                x_pred = x_pred + weights_m(i) * sigma_pred(:,i);
            end

            % Calculate predicted covariance
            P_pred = self.P;
            for i = 1:n_sigma
                diff = sigma_pred(:,i) - x_pred;
                P_pred = P_pred + weights_c(i) * (diff * diff');
            end
        end

        function action = get_action(self, state)
            % xk ~ N(x̄k, Σk)
            x_current = state(:); % Ensure column vector
            self.state = x_current';
            
            self.P = eye(self.n_sigma_states) * 0.001;

            % Initialize variables for rollouts
            self.rollouts_states = zeros(self.n_samples, self.horizon+1, self.n_states);
            self.rollouts_costs = zeros(self.n_samples, 1);

            % Generate random control input disturbance
            delta_u_vel = normrnd(0, self.cov(1), [self.n_samples, self.horizon]);
            delta_u_steer = normrnd(0, self.cov(2), [self.n_samples, self.horizon]);

            % Apply steering constraints
            delta_u_steer(delta_u_steer > 0.5) = 0.5;
            delta_u_steer(delta_u_steer < -0.5) = -0.5;

            delta_u = zeros(2, self.n_samples, self.horizon);
            delta_u(1,:,:) = delta_u_vel;
            delta_u(2,:,:) = delta_u_steer;

            n_sigma_points = 2*self.n_sigma_states + 1;
            n_sigma_batches = floor(self.n_samples / n_sigma_points);

            for batch = 1:n_sigma_batches
                current_pose = self.state(1:3)';
                [sigma_points_pose, ~, ~] = self.generateSigmaPoints(current_pose, self.P);

                for sigma_idx = 1:n_sigma_points
                    k = (batch - 1) * n_sigma_points + sigma_idx;
                    if k > self.n_samples
                        break;
                    end

                    rollout_state = [sigma_points_pose(:, sigma_idx); self.state(4:5)'];
                    total_cost = 0;

                    self.rollouts_states(k, 1, :) = rollout_state;

                    for i = 1:self.horizon
                        control_input = self.control_sequence(:,i) + delta_u(:,k,i);

                        rollout_state = self.model.step(control_input, self.dt, rollout_state);

                        self.rollouts_states(k, i+1, :) = rollout_state;

                        cost = self.cost_function(rollout_state, control_input, delta_u(:, k, i));
                        total_cost = total_cost + cost;
                    end
                    self.rollouts_costs(k) = total_cost;
                end
            end
            
            S_normalized = self.rollouts_costs - min(self.rollouts_costs);
            
            for i = 1:self.horizon
                delta_u_step = squeeze(delta_u(:,:,i))'; % n_samples x 2
                self.control_sequence(:,i) = self.control_sequence(:,i) + ...
                    self.total_entropy(delta_u_step, S_normalized);
            end

            % Apply constraints
            self.control_sequence(1, self.control_sequence(1,:) > self.max_vel) = self.max_vel;
            self.control_sequence(1, self.control_sequence(1,:) < -self.max_vel) = -self.max_vel;
            self.control_sequence(2, self.control_sequence(2,:) > self.max_steer) = self.max_steer;
            self.control_sequence(2, self.control_sequence(2,:) < -self.max_steer) = -self.max_steer;

            % Select control action
            self.optimized_control_sequence = self.control_sequence;
            action = self.control_sequence(:,1);

            % Shift control sequence
            self.control_sequence = [self.control_sequence(:,2:end), [0; 0]];
        end

        %% Cost

        function cost = cost_function(self, state, u, du)
            state_cost = self.risk_sensitive_state_cost(state);
            control_cost = self.control_cost_function(u, du);

            cost = state_cost + control_cost;
        end

        function cost = risk_sensitive_state_cost(self, state)
            obstacle_cost = self.obstacle_cost_function(state);
            heading_cost = self.heading_cost_function(state);
            distance_cost = self.risk_sensitive_distance_cost(state);

            cost = obstacle_cost + heading_cost + distance_cost;
        end

        function cost = risk_sensitive_distance_cost(self, state)
            % (15): qrs(X_k^(i), Σk)
            
            gamma = 1.0; % Risk sensitivity parameter (γ > 0: risk-averse)
            Q_matrix = diag([100, 100]); % Base weighting matrix Q
            
            % (Σk)
            Sigma_pos = self.P(1:2, 1:2); % 2x2 position covariance
            
            % (15): qrs(X_k^(i), Σk) = (1/γ)log det(I + γQΣk) + ||X_k^(i) - xf||²_Qrs
            
            % First term: (1/γ)log det(I + γQΣk)
            uncertainty_term = 0;
            try
                det_term = det(eye(2) + gamma * Q_matrix * Sigma_pos);
                if det_term > 0
                    uncertainty_term = (1/gamma) * log(det_term);
                end
            catch
                uncertainty_term = 0;
            end
            
            % Second term: ||X_k^(i) - xf||²_Qrs where Qrs = (Q^-1 + γΣk)^-1  
            try
                Q_rs = inv(inv(Q_matrix) + gamma * Sigma_pos);
            catch
                Q_rs = Q_matrix; % Fallback to original Q
            end
            
            % Calculate tracking error
            goal_pos = self.goal(1:2);     % Goal position [xf, yf]
            robot_pos = state(1:2);        % Current robot position X_k^(i)
            pos_diff = goal_pos - robot_pos;
            tracking_term = pos_diff * Q_rs * pos_diff';
            
            % (15)
            cost = uncertainty_term + tracking_term;
        end

        function cost = heading_cost_function(self, state)
            weight = 50;
            pow = 2;
            cost = weight * abs(self.get_angle_diff(self.goal(3), state(3)))^pow;
        end

        function cost = control_cost_function(self, u, du)
            cost = (1-1/self.nu)/2 * du' * self.R * du + u' * self.R * du + 1/2 * u' * self.R * u;
        end

        function obstacle_cost = obstacle_cost_function(self, state)
            if isempty(self.obstacles)
                obstacle_cost = 0;
                return
            end

            robot_pos = state(1:2);
            if size(robot_pos, 1) > size(robot_pos, 2)
                robot_pos = robot_pos';
            end
            
            obstacle_pos = self.obstacles(:,1:2);
            diff = robot_pos - obstacle_pos;
            distance_to_obstacle = sqrt(sum(diff.^2, 2));
            
            [min_dist, min_dist_idx] = min(distance_to_obstacle);
            collision_threshold = self.obstacles(min_dist_idx,3) + self.robot_radius;

            if min_dist <= collision_threshold
                hit = 1;
            else
                hit = 0;
            end

            % obstacle_cost = 550 * exp(-min_dist/10) + 1e5 * hit;
            obstacle_cost = 750 * exp(-min_dist/10) + 1e6 * hit;
        end

        function value = total_entropy(self, du, trajectory_cost)
            exponents = exp(-1/self.lambda * trajectory_cost);
            
            value = zeros(2, 1);
            sum_exp = sum(exponents);
            
            for i = 1:2
                value(i) = sum(exponents .* du(:,i)) / sum_exp;
            end
        end

        %% plot
        
        function plot_rollouts(self, fig)
            if ~isempty(self.rollouts_plot_handle)
                for i = 1:length(self.rollouts_plot_handle)
                    try
                        if ishghandle(self.rollouts_plot_handle(i))
                            delete(self.rollouts_plot_handle(i));
                        end
                    catch
                        % Handle already deleted or invalid
                    end
                end
                self.rollouts_plot_handle = [];
            end

            figure(fig);
            hold on;
            
            if isempty(self.rollouts_costs) || all(self.rollouts_costs == 0)
                return;
            end
            
            cost_range = max(self.rollouts_costs) - min(self.rollouts_costs);
            if cost_range == 0
                costs = zeros(size(self.rollouts_costs));
            else
                costs = (self.rollouts_costs - min(self.rollouts_costs)) / cost_range;
            end
            [~, min_idx] = min(self.rollouts_costs);
            
            for i = 1:self.n_samples
                if i == min_idx
                    color = [0, 1, 1];
                else
                    color = [1-costs(i), 0, 0.2];
                end
                x_traj = squeeze(self.rollouts_states(i,:,1));
                y_traj = squeeze(self.rollouts_states(i,:,2));
                self.rollouts_plot_handle(end+1) = plot(x_traj, y_traj, '-', 'Color', color);
            end
            
            % Plot uncertainty ellipses
            % self.plot_uncertainty_ellipses();
            
            % Rollout of selected trajectory
            if ~isempty(self.state) && ~isempty(self.optimized_control_sequence)
                states = zeros(self.n_states, self.horizon+1);
                states(:,1) = self.state;
                
                for i = 1:self.horizon
                    states(:,i+1) = self.model.step(self.optimized_control_sequence(:,i), self.dt, states(:,i));
                end
                self.rollouts_plot_handle(end+1) = plot(states(1,:), states(2,:), '--', 'Color', [0,1,0], 'LineWidth', 2);
            end
        end
        
      
    end
    
    methods(Static)
        function angle = get_angle_diff(angle1, angle2)
            angle_diff = angle1 - angle2;
            angle = mod(angle_diff + pi, 2*pi) - pi;
        end
    end
end