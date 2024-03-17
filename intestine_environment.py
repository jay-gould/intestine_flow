from lammps import lammps
import random

class IntestineEnvironment:

    def __init__(self, input_file, Vmax = 0.1, Vmin = 0.01, adiminsional_velocity = 0.5, number_steps = 20, episode_length = 220):
        #initialise lammps instance
        self.lmp = lammps()
        #clear any previous lammps code
        self.lmp.command("clear")
        #add input file
        self.input_file = input_file
        self.lmp.file(self.input_file)
        self.number_steps = number_steps
        #extract variables 
        self.L = self.lmp.extract_variable("dL", "LMP_VAR_EQUAL")
        #T - time per step
        self.T = self.lmp.extract_variable("dt", "LMP_VAR_EQUAL")*float(self.number_steps)
        #define maximum and minimum velocity for peristalsis
        self.Vmax = Vmax
        self.Vmin = Vmin
        #create a stopping variable to end the simulation when training
        self.is_done = False
        #set episode length
        self.episode_length = episode_length
        self.adimensional_velocity = adiminsional_velocity

    def reset(self):
        #clear any previous lammps code
        self.lmp.command("clear")
        #add input file
        self.lmp.file(self.input_file)
        #change viscosity at random
        visco_list = [0.005, 0.05, 0.5] 
        viscosity = random.choice(visco_list)
        viscosity_command = "variable c equal " + str(viscosity)
        self.lmp.command(viscosity_command)
        #reset variables
        self.intestine_stretch = None
        self.fluid_centre_of_mass = None
        self.is_done = False
        self.previous_stretch = 0.0

        #start contraction to get first observation
        self.adimensional_velocity = 0.4
        vel_peri = self.Vmin + self.adimensional_velocity*(self.Vmax-self.Vmin)
        vel_peri_command = "variable vel_peri equal " + str(vel_peri)
        self.lmp.command(vel_peri_command)

        #run the simulation at that velocity for first few timesteps
        run_command = "run " + str(self.number_steps)
        self.lmp.command(run_command)

        #extract the stretch to feed into RL model as input
        self.intestine_stretch = (self.lmp.extract_compute("eps",0,0)*10000)-23

        #extract centre of mass to use in our reward function for the RL model
        self.fluid_centre_of_mass =  self.lmp.extract_compute("com",0,1)
        self.fluid_centre_of_mass_z = self.fluid_centre_of_mass[2]

        #create numpy array of observation for input
        self.observation = [self.intestine_stretch]

        return self.observation

    def run_simulation(self):
        #set the velocity of the peristalsis 
        vel_peri = self.Vmin + self.adimensional_velocity*(self.Vmax-self.Vmin)
        vel_peri_command = "variable vel_peri equal " + str(vel_peri)
        self.lmp.command(vel_peri_command)

        #run the simulation at that velocity for 'number_steps' timesteps
        run_command = "run " + str(self.number_steps)
        self.lmp.command(run_command)

        #previous stretch update
        self.previous_stretch = self.intestine_stretch
        #extract the stretch to feed into RL model as input
        self.intestine_stretch = (self.lmp.extract_compute("eps",0,0)*10000)-23
        #create observation
        self.observation = [self.intestine_stretch]

        previous_com = self.fluid_centre_of_mass_z
        #extract centre of mass to use in our reward function for the RL model
        self.fluid_centre_of_mass =  self.lmp.extract_compute("com",0,1)
        self.fluid_centre_of_mass_z = self.fluid_centre_of_mass[2]
        #transform into usable reward
        velocity_com = (self.fluid_centre_of_mass_z - previous_com)/self.T
        self.reward = velocity_com/self.L #reward in units of L

        #extract position variables for point of contraction
        new_z0_initial = self.lmp.extract_variable('z0')
        new_z1_initial = self.lmp.extract_variable('z1')
        new_z2_initial = self.lmp.extract_variable('z2')

        #make sure contraction stays in the same place
        z0_init_commmand = "variable z0_initial equal " + str(new_z0_initial)
        z1_init_commmand = "variable z1_initial equal " + str(new_z1_initial)
        z2_init_commmand = "variable z2_initial equal " + str(new_z2_initial)

        self.lmp.command(z0_init_commmand)
        self.lmp.command(z1_init_commmand)
        self.lmp.command(z2_init_commmand)

        #reset current_timestep to make sure peristalsis stays in same place
        actual_current_step = self.lmp.extract_variable('actual_current_step')
        reset_step_command = "variable current_step equal step-" + str(actual_current_step)
        self.lmp.command(reset_step_command) 
    
        #check to make sure we are within episode length
        if actual_current_step >= self.episode_length:
            self.is_done = True
        else:    
            pass
    
    def step(self, action):
        #change velocity based on NN action
        if action == 5:
            self.adimensional_velocity = 1.0
        elif action == 4:
            self.adimensional_velocity = 0.8
        elif action == 3:
            self.adimensional_velocity = 0.6
        if action == 2: 
            self.adimensional_velocity = 0.4
        elif action == 1: 
            self.adimensional_velocity = 0.2
        elif action == 0: 
            self.adimensional_velocity = 0.0
        
        #keep adimension_velocity in between 1 and 0
        if self.adimensional_velocity > 1.0:
            self.adimensional_velocity = 1.0
        elif self.adimensional_velocity < 0.0:
            self.adimensional_velocity = 0.0
        else:
            pass
        
        #run the simulation with the current adim velocity
        self.run_simulation()

        return self.observation, self.reward, self.is_done
    
if __name__ == "__main__":
    #(number of actions + 1) * number of steps = episode length
    action_list = [2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1] #actions to vary speed
    Vmax , Vmin = 0.2, 0.01 #max and min velocity of the peristalsis


    input_file = r"C:\Users\gould\Desktop\PhD\pipe-flow\pass_to_python\intestine_flow.lmp"
    it_environment = IntestineEnvironment(input_file = input_file, Vmax = Vmax, Vmin = Vmin)

    initial_strech = it_environment.reset()

    stretch_list = []
    reward_list = []
    is_done_list = []

    for action in action_list:
        intestine_stretch, reward, is_done = it_environment.step(action)

        stretch_list.append(intestine_stretch)
        reward_list.append(reward)
        is_done_list.append(is_done)

    print("Stretch list: ", stretch_list)
    print("Reward List: ", reward_list)
    print("Is done: ", is_done_list)