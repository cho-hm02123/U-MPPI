%%
close all
clear
clc

%% Parameters
n_rollout = 2499;           % Number of rollout trajectories (M in U-MPPI)
horizon = 25;               % Prediction horizon (N in U-MPPI)
lambda = 10;                % Temperature parameter
nu = 500;                   % Exploration variance
R = diag([1,5]);            % Control weight matrix
cov = [1,0.4];              % Variance of control inputs disturbance
dt = 0.1;                   % Time step

init_pose = zeros(1,5);     % Initial pose [x, y, phi, v, steer]
goal_pose = [6,6,0];
goal_tolerance = 0.3;

%% Setup Environment - Obstacles
% n_obstacles = 40;
% obstacles = [rand(n_obstacles,2)*4+1, 0.2*ones(n_obstacles,1)];

% 
o = load("ob1.mat");
obstacles = o.obstacles;
n_obstacles = size(obstacles);

%% Initialize Dynamics & Unscented MPPI Controller
v_dynamics = VehicleModel();
car = VehicleModel();
controller = UnscentedMPPIController(lambda, cov, nu, R, horizon, n_rollout, car, dt, goal_pose, obstacles);

%% Visualization
fig = figure;
hold on
axis equal
xlim([-0.5 + min(init_pose(1), goal_pose(1)), 0.5 + max(init_pose(1), goal_pose(1))]);
ylim([-0.5 + min(init_pose(2), goal_pose(2)), 0.5 + max(init_pose(2), goal_pose(2))]);
plot_pose(init_pose,'bo');
plot_pose(goal_pose, 'ro');
plot_obstacles(obstacles);

%% Data recording
max_steps = 1000;
time_data = [];
input_velocity = [];
input_steering = [];
actual_velocity = [];
actual_steering = [];
collision_times = [];
collision_positions = [];

% Additional data for Unscented Transform analysis
state_estimates = [];
state_covariances = []; % Will be 3D array: (time_step, state_dim, state_dim)

%% Control loop
car_pose = init_pose;
goal_reached = false;
collision_detected = false;
step_count = 0;

for i = 1:max_steps
    step_count = i;
    current_time = (i-1) * dt;

    % Get control action using Unscented MPPI
    action = controller.get_action(car_pose);

    % Record data
    time_data(end+1) = current_time;
    input_velocity(end+1) = action(1);
    input_steering(end+1) = action(2);

    % Store state estimates and covariances
    state_estimates(end+1,:) = controller.state;
    state_covariances(end+1,:,:) = controller.P;

    % Update vehicle state
    car_pose = car.step(action, dt, car_pose);
    
    actual_velocity(end+1) = car_pose(4);
    actual_steering(end+1) = car_pose(5);

    % Check for collision
    if check_collision(car_pose, obstacles)
        collision_detected = true;
        collision_times(end+1) = current_time;
        collision_positions(end+1,:) = car_pose(1:2);
        fprintf('Collision detected at time %.2f at position (%.2f, %.2f)\n', current_time, car_pose(1), car_pose(2));
    end
    
    % Check if goal is reached
    distance_to_goal = norm(car_pose(1:2) - goal_pose(1:2));
    if distance_to_goal <= goal_tolerance
        goal_reached = true;
        fprintf('Goal reached at time %.2f! Distance to goal: %.3f\n', current_time, distance_to_goal);
        break;
    end

    % Visualization
    if mod(i,2) == 0
        controller.plot_rollouts(fig);
        exportgraphics(gcf, 'unscented_mppi_animation2.gif', 'Append', true);
        drawnow;
    end

    plot_pose(car_pose, 'go');

    % Safety check
    if distance_to_goal > 20
        fprintf('Robot moved too far from goal. Stopping simulation.\n');
        break;
    end
end

%% Results
if goal_reached
    fprintf('Simulation completed successfully! Goal reached in %d steps (%.2f seconds).\n', step_count, step_count*dt);
else
    fprintf('Simulation ended without reaching goal. Final distance: %.3f\n', norm(car_pose(1:2) - goal_pose(1:2)));
end

if collision_detected
    fprintf('Total collisions detected: %d\n', length(collision_times));
end

%% Plot Results
plot_unscented_mppi_results(time_data, input_velocity, input_steering, ...
                           actual_velocity, actual_steering, ...
                           state_estimates, state_covariances);

%% Utility Functions
function plot_obstacles(obstacles)
    for i = 1:size(obstacles,1)
        r = obstacles(i,3);
        pos = [obstacles(i,[1,2])-r 2*r 2*r];
        rectangle('Position',pos, 'Curvature',[1,1], 'FaceColor','k','EdgeColor','none');
    end
end

function plot_pose(pose, style)
    x = pose(1);
    y = pose(2);
    phi = pose(3);
    plot(x,y,style);
    [delta_x, delta_y] = pol2cart(phi,0.5);
    quiver(x,y,delta_x,delta_y);
end

function collision = check_collision(pose, obstacles)
    collision = false;
    if isempty(obstacles)
        return;
    end
    
    robot_position = pose(1:2); % Keep as row vector
    obstacle_positions = obstacles(:,1:2); % n_obstacles x 2
    
    % Calculate distances using broadcasting
    diff = robot_position - obstacle_positions; % n_obstacles x 2
    distances = sqrt(sum(diff.^2, 2)); % n_obstacles x 1
    obstacle_radii = obstacles(:,3);
    collision = any(distances <= obstacle_radii);
end

function plot_unscented_mppi_results(time_data, input_vel, input_steer, actual_vel, actual_steer, state_estimates, state_covariances)
    figure('Name', 'Unscented MPPI Results', 'Position', [100, 100, 1400, 1000]);
    
    % Plot 1: Velocity comparison
    subplot(2,3,1);
    plot(time_data, input_vel, 'r-', 'LineWidth', 2, 'DisplayName', 'Input Velocity');
    hold on;
    plot(time_data, actual_vel, 'b--', 'LineWidth', 2, 'DisplayName', 'Actual Velocity');
    xlabel('Time (s)');
    ylabel('Velocity (m/s)');
    title('Velocity Comparison');
    legend('Location', 'best');
    grid on;
    
    % Plot 2: Steering comparison
    subplot(2,3,2);
    plot(time_data, input_steer, 'r-', 'LineWidth', 2, 'DisplayName', 'Input Steering');
    hold on;
    plot(time_data, actual_steer, 'b--', 'LineWidth', 2, 'DisplayName', 'Actual Steering');
    xlabel('Time (s)');
    ylabel('Steering (rad)');
    title('Steering Comparison');
    legend('Location', 'best');
    grid on;
    
    % Plot 3: State estimates vs actual
    subplot(2,3,3);
    plot(time_data, state_estimates(:,1), 'r-', 'LineWidth', 2, 'DisplayName', 'Estimated X');
    hold on;
    plot(time_data, state_estimates(:,2), 'b-', 'LineWidth', 2, 'DisplayName', 'Estimated Y');
    xlabel('Time (s)');
    ylabel('Position (m)');
    title('State Estimates');
    legend('Location', 'best');
    grid on;
    
    % Plot 4: Position uncertainty (standard deviation)
    subplot(2,3,4);
    pos_std_x = sqrt(squeeze(state_covariances(:,1,1)));
    pos_std_y = sqrt(squeeze(state_covariances(:,2,2)));
    plot(time_data, pos_std_x, 'r-', 'LineWidth', 2, 'DisplayName', 'X Std Dev');
    hold on;
    plot(time_data, pos_std_y, 'b-', 'LineWidth', 2, 'DisplayName', 'Y Std Dev');
    xlabel('Time (s)');
    ylabel('Standard Deviation (m)');
    title('Position Uncertainty');
    legend('Location', 'best');
    grid on;
    
    % Plot 5: Heading uncertainty
    subplot(2,3,5);
    heading_std = sqrt(squeeze(state_covariances(:,3,3)));
    plot(time_data, heading_std, 'g-', 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel('Standard Deviation (rad)');
    title('Heading Uncertainty');
    grid on;
    
    % Plot 6: Velocity uncertainty
    subplot(2,3,6);
    vel_std = sqrt(squeeze(state_covariances(:,4,4)));
    steer_std = sqrt(squeeze(state_covariances(:,5,5)));
    plot(time_data, vel_std, 'r-', 'LineWidth', 2, 'DisplayName', 'Velocity Std');
    hold on;
    plot(time_data, steer_std, 'b-', 'LineWidth', 2, 'DisplayName', 'Steering Std');
    xlabel('Time (s)');
    ylabel('Standard Deviation');
    title('Control State Uncertainty');
    legend('Location', 'best');
    grid on;
    
    sgtitle('Unscented MPPI Performance Analysis', 'FontSize', 16, 'FontWeight', 'bold');
end