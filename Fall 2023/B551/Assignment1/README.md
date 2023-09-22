# Assignment - 0

## Report

### Problem1

To my understanding this is a basic search problem and can be easily solved by Breadth First Search(BFS) approach. The code also provide most of the solution. The fixes I made to the code follows,
- Fixing the code to use queue based location exploration
- Added a visited_map to store all the location that has been visited previously
    - This was actually the cause that the program was running endlessly in a loop
- Added the section to store and check whether a location has been previously visited and then add it to the fringe if it an entire new location
- To calculate the distance, I used the given distance storing functionality and add distance on top of it as each location was being explored
- To get the final distance where we found our goal, instead of returning the distance I incremented with 1 as that way we will consider moving to the distance. Otherwise it will not be possible and we will always get distance one less than the actual
- To find the traversing distance, I used the move data and added the relevant move from the current move.
    For example, if the current move is (1,1) and the next move found is (1,2) then along with the distance I added the corresponding move which is "R" in this scenario. And, like distance the moves is appended on top of another giving the final move all the required performed move. So, by using this I was able to store the moving data in the BFS map. Also, I had to overload the `moves` function with an additional parameter `displaced_loc` which is for storing and adding related moves. Furthermore, I tried running the file in the silo server and it ran perfectly so hopefully this is okay!
- Finally, the program utilizes the bfs algorithm to find the goal position that is the portal. However, since this algorithm provides functionality to traverse all the location possible visiting so if after visiting all the position the portal is not found then I have simply returned `-1` denoting that there is no goal position.

### Problem2

For this problem, although it is pretty much similar to the N_Queen problem the addition of wall `X` complicated the solution. So instead of following the traditional approach to solve NQueen problem, a NP-Hard problem I tried to work with the given turret positions. At first, I though going through all the rows, columns and diagonal points to find whether there is any conflicting turrets. However, that prove to be a much complicated approach so instead I followed the below procedure,
- In my approach, I utilized the `add_turrets` function and then implemented a function where it gives me the position of the turrets. The reason I didn't modify the mentioned function because I wanted to work as intended and not yeilds any more result
- Then using the turret position, I implemented some functions to check whether there is a confliction with the turrets
    - The function I implemented check for every turret and traverse from its relative position to its corresponding row, column and diagonal way
    - For every side it checks, it any side yields a conflict then the function yields a False result
    - The false result denotes that there is a conflict with the current castle map turret positions and it could be with any turret
    - I will only consider the caste map that has turrets with no confliction
- Then, if the castle map has a no-confliction map then it gets added to the fringe as a successfull map and further put to test whether the map has reached our goal turret count
- Finally, the algorithm proceed to find solution until no new castle map can be created with new turret position
    - If after all the turret placing we are unable to find a solution then the function returns the castle map and empty string as it can then be identified as a not possible solution
- Additionally, it has been assumed that there will always be a position to place the turret at the start of the program and hence no check has been implemented to ensure or check any map that has no position for placing turret

[Note: I have splitted by conflict check functions into 8 different function as it will check for every suggested direction individually and will return if any of them have any conflictions. The confliction will not counted if it encounters an `X` as it simply means that in that particular way the turret will not face any conflictions. Also, I have only included the castle map that has no conflicted turret resulting in a much reduced space complexity.]
