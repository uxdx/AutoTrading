"""Comment Line Interface 실행문
"""


# Import the click module that was just downloaded.
import click


# Makes the function main a designated group for commands. Allowing multiple functions to be tied to main.
@click.group()
# Function that represents a group since there is no command decorator.
def main():
	# Creating an empty function since we need to have this exist so we can use it as a group.
	pass


# Uses the "main" group to indicate that this function is a command of that group.
@main.command()
@click.option(
        "--path",
        "-p",
        default='./',
        help="Set a dataset file path."
)
def create(path):
	"""
	Create dataset in specific path as options
	"""

	# Simple print statement to the terminal.

	click.echo(path)


# Uses the "main" group to indicate that this function is a command of that group.
@main.command()
# Function that represents a command. The function name is the command name.
def delete():
	"""
	Prints the value "World!" to the terminal.
	"""
	# Simple print statement to the terminal.
	click.echo("World!")


if __name__ == "__main__":
	main()