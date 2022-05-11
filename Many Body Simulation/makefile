create: sim graphics
	@echo "\n\n Run following commands for 1 thread"
	@echo "    > ./sim trajectory1.dat 1"
	@echo "    > ./graphics trajectory1.dat"
	@echo "\n Please trajectory filename with .dat extension. Do similarly for different number of threads.."

sim: many-body-sim-program.cpp 
	g++ -fopenmp many-body-sim-program.cpp -o sim    

graphics: graphics-program.cpp   
	g++ -fopenmp graphics-program.cpp -lglfw -lGL -lm -lX11 -lpthread -lXi -lXrandr -ldl -lglut -lGL -lGLU -o graphics

clean: 
	rm sim graphics *.dat