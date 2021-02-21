# viminidb

Viminidb is a simple database manager running in py. Important disclaimer:
**Viminidb starts up, loads whole tables, and does its whole spiel on every incoming query.** The reason is simple. Its primary purpose is to manage tiny personal tables from the terminal.

So again: Very simple, not optimized, use with caution, etc.

## Running viminidb

Running viminidb

```
python viminidb.py -h
```

Running a sample hello world query

```
python viminidb.py "create helloworld lines[str] >> insert 'hello' >> insert 'world'"
```

## Storage

Everything is stored in folder .viminidbstorage that is placed right next to the script. This ensures that no matter where the script is running from, it uses the same tables and scripts.

### Tables

Tables are stored in .viminidbstorage/tables. They are simple csv files with ';' delimiter.

### Scripts

Scripts are stored as mndb files and contain only plain text.

## Domains

1. int (1, 2, -3, ...)
1. float (1.1, 2.3, 4.5, -1.0, ...)
1. bool (True, False)
1. str ('hello') - note that you can only use ' not "
1. date ('2020-10-20', '1099/10/20', '31.12.1999') - note that the best way of inputing date is string meaning using '
1. datetime ('2020-12-31 10:00:00', '2020-12-31 10:00', '2020-12-31 10')
1. null

## The query structure

Each viminidb query is an array of commands linked either using (.) or (>>).

### after operator (.)

Links functions "from the right to the left", meaning expression on the right is executed first, and its result is forwarded to the function on the left.

### then operator (>>)

Links function in intuitive order (from left to right). First, the expression on the left is executed, its result is then forwarded to the function on the right.

## Different functions

The first important note is that every function has one hidden argument, a table. Meaning every function expression results in a function that takes a table and produces another table.

### nothing

Does nothing, grabs the input table, and returns it.

### create _table_name_ _col_def1_ _col_def2_ _col_def3_ ...

Throws the input table away, creates a new table with name _table_name_ using the column definitions.

Column definition can be either very simple only containing the name and type: `name[str]` or complex tuples `(id[int], 'key', '%')`.

### load _table_name_

Throws input table away and loads and returns the table with name _table_name_

### save _table_name_?

Saves the input table to the storage and returns it. If the table name is provided, it is saved under that name, otherwise under its own name.

### drop _table_name_?

If a table name is provided, drops the table with said name, otherwise drops the input table, either-way returns None.

### rename _table_name_

Renames the input table and returns it

### print

Prints the table to the terminal and returns it.

### project _row_fun_1_ _row_fun_2_ ...

Returns a new table where every row is mapped to a new tuple, so that _i_-th value is the result of _row_fun_i_ being applied on the said row. The columns have names col_i.

### select _predicate_

Returns a new table with the same columns as the input table but only copies the rows that satisfy predicate _predicate_ (Meaning applying _predicate_ on row produces True).

### insert _tuple_

Inserts the tuple into the input table and returns it.

### update _predicate_ _row_fun_1_ _row_fun_2_ ...

If row from the input table satisfies the predicate, its values are updated using the row functions so that _i_-th value of the new tuple is the result of applying _row_fun_i_ on the old row.

### remove _predicate_

Removes rows from the input table that satisfy predicate _predicate_ and returns the input table.

### cartesian _table_expr_1_ _table_expr_2_ _predicate_ ?

_table_expr_1_ and _table_expr_2_ are applied on the input table producing two new tables. These tables have their columns prefixed with their names. Cartesian product is applied on these tables producing the output table that is returned.

(The predicate is used during the product to limit the number of combinations)

### sort _col_name_1_ _col_name_2_ _col_name_3_ ...

Sorts the input table lexicographically on the columns from arguments and returns it.

### script _script_name_

Runs the script on the input table and returns the result.

## Row functions and predicates

Row functions are functions that take _row_ and _table_ and return single. Let us imagine we have the following query:

```
"create nats n[int] >> insert 0 >> insert 1 >> insert 2 >> insert 3"
```

that results in the following table:

```
nats
┌─────┐
│n    │
│[int]│
╞═════╡
│0    │
├─────┤
│1    │
├─────┤
│2    │
├─────┤
│3    │
└─────┘
```

Let us create a projection that has two columns, where the first contains the original number, second the number multiplied by three, third boolean if the multiple is equal to 6.

```
create nats n[int] >> insert 0 >> insert 1 >> insert 2 >> insert 3 >> print >> project n (n * 3) (n * 3 == 6)
```

This query results in:

```
 projection_table
┌─────┬─────┬──────┐
│col_0│col_1│col_2 │
│[int]│[int]│[bool]│
╞═════╪═════╪══════╡
│0    │0    │False │
├─────┼─────┼──────┤
│1    │3    │False │
├─────┼─────┼──────┤
│2    │6    │True  │
├─────┼─────┼──────┤
│3    │9    │False │
└─────┴─────┴──────┘
```

Note that the original query is fully contained in the new query, and thus the only important part is the

```
project n (n * 3) (n * 3 == 6)
```

vimindb currently supports following operators:

- (.), (>>) - described in query section
- (,) - creates tuple from the left and the right side (there can't be a tuple of tuples)
- (=>), (!>) - 'in' and 'not in' operator `a => t` checks if value a is in the table t
- (&&), (||) - simple 'and' and 'or'
- (==), (!=), (<=), (>=), (<), (>) - you get it
- (+), (-), (\*) - function as you would expect
- (/), (//) - float division (always float) and int division (always int)
- (^) - power `a ^ b` is a to the power of b
- (%) - modulo
  Note that viminidb has somewhat weak typing, meaning the database will try its hardest to interpret the value as the value it needs, but don't try to do anything crazy.

## Scripting system

Any query as a plain text saved as .mndb in the scripts folder can be run using

```
viminidb -us <script_name>
```

If you run any query using -rs <script_name> that query will be and save with that name

```
viminidb -rs "hello_world" "create helloworld lines[str] >> insert 'hello' >> insert 'world'"
```

The script will be saved as hello_world.mndb in the scripts folder

```
viminidb -us "hello_world"
```

will result in

```
┌─────┐
│lines│
│[str]│
╞═════╡
│hello│
├─────┤
│world│
└─────┘
```

## Queries with parameters

Positional arguments given to viminidb after the query will be taken as arguments. You can catch these arguments using using $n or $n|type$ where n is the position of the argument (indexing from 1).

Running

```
viminidb -rs "hello_world" "create helloworld lines[str] >> insert @1|str@ >> insert @2|str@" hello world
```

will result in

```
┌─────┐
│lines│
│[str]│
╞═════╡
│hello│
├─────┤
│world│
└─────┘
```

Note that the |str is important as it is not a simple text replacement. If you only replaced @1 with hello, it will be taken as an attribute name and not a value. Thus it will not be compatible with the insert function. The |str inserts '' around your input.

Parameters of course work also with scripts note that if you use a script, the very first argument is already an additional argument (normally, the first argument is the query)
