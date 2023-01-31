from ic_routing_board_generation.ic_routing.route import Route
from ic_routing_board_generation.interface.board_generator_interface import \
    BoardGenerators

if __name__ == '__main__':
    router = Route(instance_generator_type=BoardGenerators.DUMMY,
                         rows=8,
                         cols=8,
                         num_agents=3,
                         step_limit=30)
    router.route(time_steps=50, fps=15, **router.__dict__)
