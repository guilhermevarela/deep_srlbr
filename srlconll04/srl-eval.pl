#! /usr/bin/perl 

##################################################################
#
#  srl-eval.pl : evaluation program for the CoNLL-2004 Shared Task
#
#  Authors : Xavier Carreras and Lluis Marquez
#  Contact : carreras@lsi.upc.es
#
#  1 March 2004
#
#  If you find BUGS, please get in touch ASAP with the authors.
#
#  This program is implemented in Perl, object oriented, with each
#  object embedded in the script. If you would like to use or extend
#  the code, we can provide you with a suite of Perl modules, with
#  each object as a separate module. Contact the authors.  
#
##################################################################

use strict; 


############################################################
#  A r g u m e n t s   a n d   H e l p 

use Getopt::Long; 
my %options;
GetOptions(\%options, 
           "latex"     # latex output
           ); 


my $script = "srl-eval.pl"; 
my $help = << "end_of_help;";
Usage:   srl-eval.pl [-latex] <gold props> <predicted props>
end_of_help;


############################################################
#  M A I N   P R O G R A M 


my $ns = 0;        # number of sentence
my $ntargets = 0;  # number of target verbs
my %E;             # evaluation results

# the categories which, as keys, have a true value in the following hash are excluded from evaluation
my %excluded = ( V => 1); 

##

# open files

if (@ARGV != 2) {
    print $help;
    exit; 
}

my $goldfile = shift @ARGV;
my $predfile = shift @ARGV;

if ($goldfile =~ /\.gz/) {
    open GOLD, "gunzip -c $goldfile |" or die "$script: could not open gzipped file of gold props ($goldfile)! $!\n"; 
}
else {
    open GOLD, $goldfile or die "$script: could not open file of gold props ($goldfile)! $!\n"; 
}
if ($predfile =~ /\.gz/) {
    open PRED, "gunzip -c $predfile |" or die "$script: could not open gzipped file of predicted props ($predfile)! $!\n"; 
}
else {
    open PRED, $predfile or die "$script: could not open file of predicted props ($predfile)! $!\n"; 
}


##
# read and evaluate propositions, sentence by sentence

my $s = SRL::sentence->read_props($ns, GOLD => \*GOLD, PRED => \*PRED); 

while ($s) {

    my $prop; 

    my (@G, @P, $i); 
    
    map { $G[$_->position] = $_ } $s->gold_props; 
    map { $P[$_->position] = $_ } $s->pred_props; 
    
    for($i=0; $i<@G; $i++) {
	my $gprop = $G[$i]; 
	my $pprop = $P[$i]; 

	if ($pprop and !$gprop) {	    
	    print STDERR "WARNING : sentence $ns : verb ", $pprop->verb, " at position ", $pprop->position, 
	    " : found predicted prop without its gold reference! Skipping prop!\n"; 
	}
	elsif ($gprop) {
	    if (!$pprop) {
		print STDERR "WARNING : sentence $ns : verb ", $gprop->verb, " at position ", $gprop->position, 
		" : missing predicted prop! Counting all arguments as missed!\n"; 
		$pprop = SRL::prop->new($gprop->verb, $gprop->position); 
	    }
	    elsif ($gprop->verb ne $pprop->verb) {
		print STDERR "WARNING : sentence $ns : props do not match : expecting ", 
		$gprop->verb, " at position ", $gprop->position, 
		", found ", $pprop->verb, " at position ", $pprop->position, "! Counting all gold arguments as missed!\n";
		$pprop = SRL::prop->new($gprop->verb, $gprop->position);
	    }
       	
	    $ntargets++;
	    my %e = evaluate_proposition($gprop, $pprop); 


	    # Update global evaluation results

	    $E{ok} += $e{ok};
	    $E{op} += $e{op};
	    $E{ms} += $e{ms};
	    $E{ptv} += $e{ptv};

	    my $t; 
	    foreach $t ( keys %{$e{T}} ) {
		$E{T}{$t}{ok} += $e{T}{$t}{ok};
		$E{T}{$t}{op} += $e{T}{$t}{op};
		$E{T}{$t}{ms} += $e{T}{$t}{ms};
	    }
	    foreach $t ( keys %{$e{E}} ) {
		$E{E}{$t}{ok} += $e{E}{$t}{ok};
		$E{E}{$t}{op} += $e{E}{$t}{op};
		$E{E}{$t}{ms} += $e{E}{$t}{ms};
	    }
	}
    }

    $ns++; 
    $s = SRL::sentence->read_props($ns, GOLD => \*GOLD, PRED => \*PRED); 

}


# Print Evaluation results
my $t; 

if ($options{latex}) {
    print '\begin{table}[t]', "\n"; 
    print '\centering', "\n"; 
    print '\begin{tabular}{|l|r|r|r|}\cline{2-4}',  "\n"; 
    print '\multicolumn{1}{l|}{}', "\n"; 
    print '           & Precision & Recall & F$_{\beta=1}$', '\\\\', "\n", '\hline', "\n";  #'

    printf("%-10s & %6.2f\\%% & %6.2f\\%% & %6.2f\\\\\n", "Overall", precrecf1($E{ok}, $E{op}, $E{ms})); 
    print '\hline', "\n"; 

    foreach $t ( sort keys %{$E{T}} ) {
	printf("%-10s & %6.2f\\%% & %6.2f\\\% & %6.2f\\\\\n", $t, precrecf1($E{T}{$t}{ok}, $E{T}{$t}{op}, $E{T}{$t}{ms})); 
    }
    print '\hline', "\n"; 
    
    if (%excluded) {
	print '\hline', "\n"; 
	foreach $t ( sort keys %{$E{E}} ) {
	    printf("%-10s & %6.2f\\%% & %6.2f\\\% & %6.2f\\\\\n", $t, precrecf1($E{E}{$t}{ok}, $E{E}{$t}{op}, $E{E}{$t}{ms})); 
	}
	print '\hline', "\n"; 
    }
    
    print '\end{tabular}', "\n"; 
    print '\end{table}', "\n"; 
}
else {
    printf("Number of Sentences    :      %6d\n", $ns); 
    printf("Number of Propositions :      %6d\n", $ntargets); 
    printf("Percentage of perfect props : %6.2f\n",($ntargets>0 ? 100*$E{ptv}/$ntargets : 0)); 
    print "\n"; 
    
    printf("%10s   %6s  %6s  %6s   %6s  %6s  %6s\n", "", "corr.", "excess", "missed", "prec.", "rec.", "F1"); 
    print "------------------------------------------------------------\n"; 
    printf("%10s   %6d  %6d  %6d   %6.2f  %6.2f  %6.2f\n", 
	   "Overall", $E{ok}, $E{op}, $E{ms}, precrecf1($E{ok}, $E{op}, $E{ms})); 
#    print "------------------------------------------------------------\n"; 
    print "----------\n"; 
    
#    printf("%10s   %6d  %6d  %6d   %6.2f  %6.2f  %6.2f\n", 
#	   "all - {V}", $O2{ok}, $O2{op}, $O2{ms}, precrecf1($O2{ok}, $O2{op}, $O2{ms})); 
#    print "------------------------------------------------------------\n"; 
    
    foreach $t ( sort keys %{$E{T}} ) {
	printf("%10s   %6d  %6d  %6d   %6.2f  %6.2f  %6.2f\n", 
	       $t, $E{T}{$t}{ok}, $E{T}{$t}{op}, $E{T}{$t}{ms}, precrecf1($E{T}{$t}{ok}, $E{T}{$t}{op}, $E{T}{$t}{ms})); 
    }
    print "------------------------------------------------------------\n"; 

   foreach $t ( sort keys %{$E{E}} ) {
	printf("%10s   %6d  %6d  %6d   %6.2f  %6.2f  %6.2f\n", 
	       $t, $E{E}{$t}{ok}, $E{E}{$t}{op}, $E{E}{$t}{ms}, precrecf1($E{E}{$t}{ok}, $E{E}{$t}{op}, $E{E}{$t}{ms})); 
    }
    print "------------------------------------------------------------\n"; 
}


# end of main program
#####################

############################################################
#  S U B R O U T I N E S


# evaluates a predicted proposition wrt the gold correct proposition
# returns a hash with the following keys
#   ok  :  number of correctly predicted args
#   ms  :  number of missed args
#   op  :  number of over-predicted args
#   T   :  a hash indexed by argument types, where
#           each value is in turn a hash of {ok,ms,op} numbers
#   E   :  a hash indexed by excluded argument types, where
#           each value is in turn a hash of {ok,ms,op} numbers
sub evaluate_proposition {
    my ($gprop, $pprop) = @_;

    my $o = $gprop->discriminate_args($pprop); 
    
    my %e; 

    my $a; 
    foreach $a (@{$o->{ok}}) {
	if (!$excluded{$a->type}) {
	    $e{ok}++;
	    $e{T}{$a->type}{ok}++;
	}
	else {
	    $e{E}{$a->type}{ok}++;
	}
    }
    foreach $a (@{$o->{op}}) {
	if (!$excluded{$a->type}) {
	    $e{op}++;
	    $e{T}{$a->type}{op}++;
	}
	else {
	    $e{E}{$a->type}{op}++;
	}
    }
    foreach $a (@{$o->{ms}}) {
	if (!$excluded{$a->type}) {
	    $e{ms}++;
	    $e{T}{$a->type}{ms}++;
	}
	else {
	    $e{E}{$a->type}{ms}++;
	}
    }

    $e{ptv} = (!$e{op} and !$e{ms}) ? 1 : 0; 
    
    return %e; 
}


# computes precision, recall and F1 measures
sub precrecf1 {
    my ($ok, $op, $ms) = @_;

    my $p = ($ok + $op > 0) ? 100*$ok/($ok+$op) : 0; 
    my $r = ($ok + $ms > 0) ? 100*$ok/($ok+$ms) : 0; 

    my $f1 = ($p+$r>0) ? (2*$p*$r)/($p+$r) : 0; 

    return ($p,$r,$f1); 
}



# end of script
###############





















################################################################################
#
#   Package    s e n t e n c e
#
#   February 2004
#
#   Stores information of a sentence, namely words, chunks, clauses,
#   named entities and propositions (gold and predicted). 
#
#   Provides access methods.
#   Provides methods for reading/writing sentences from/to files in 
#    CoNLL-2004 formats. 
#
#
################################################################################


package SRL::sentence; 


sub new {
    my ($pkg, $id) = @_; 
    
    my $s = []; 

    $s->[0] = $id;      # sentence number
    $s->[1] = undef;    # words (the list or the number of words)
    $s->[2] = undef;    # chunks
    $s->[3] = undef;    # clauses
    $s->[4] = undef;    # named entities
    $s->[5] = [];       # gold props
    $s->[6] = [];       # predicted props

    return bless $s, $pkg; 
}

#-----

sub id {
    my $s = shift; 
    return $s->[0];
}

#-----

sub length {
    my $s = shift; 
    if (ref($s->[1])) {
	return scalar(@{$s->[1]}); 
    }
    else {
	return $s->[1]; 
    }
}

sub set_length {
    my $s = shift; 
    $s->[1] = shift; 
}

#-----

# returns the list of words of the sentence
# non-paranoic implementation: position [1] might be the number of words, instead of the list, but this is not checked
sub words {
    my $s = shift;
    if (@_) {
        return map { $s->[1][$_] } @_;
    }
    else {
        return @{$s->[1]};
    }
}


sub chunking {
    my $s = shift; 
    return $s->[2];
}

sub clausing {
    my $s = shift; 
    return $s->[3];
}


#-----

sub add_gold_props {
    my $s = shift;
    push @{$s->[5]}, @_;
}

sub gold_props {
    my $s = shift; 
    return @{$s->[5]};
}

sub add_pred_props {
    my $s = shift;
    push @{$s->[6]}, @_;
}

sub pred_props {
    my $s = shift; 
    return @{$s->[6]};
}


#------------------------------------------------------------
# I/O  F U N C T I O N S
#------------------------------------------------------------

# Reads a complete (words, synt, props) sentence from a stream
# Returns: the reference to the sentence object or
#          undef if no sentence found 
# The propositions in the file are stored as gold props
# For each gold prop, an empty predicted prop is created
sub read_from_stream {
    my ($pkg, $id, $fh) = @_;

    my $cols = read_columns($fh); 

    if (!@$cols) {
	return undef; 
    }
    
    my $s = $pkg->new($id); 
    
    # words and PoS
    my $words = shift @$cols;
    my $pos = shift @$cols;

    # initialize list of words
    $s->[1] = [];
    my $i;    
    for ($i=0;$i<@$words;$i++) {
	push @{$s->[1]}, SRL::word->new($i, $words->[$i], $pos->[$i]);
    }
    
    # chunks
    my $chunks = shift @$cols; 
    # initialize chunking
    $s->[2] = SRL::phrase_set->new(); 
    $s->[2]->load_IOB2_tagging(@$chunks); 

    # clauses
    my $clauses = shift @$cols; 
    # initialize clauses
    $s->[3] = SRL::phrase_set->new(); 
    $s->[3]->load_SE_tagging(@$clauses); 


    my $targets = shift @$cols; 
    if (@$cols) {
	$s->load_props($s->[5], $targets, $cols); 
    }

    # initialize predicted props
    foreach $i ( grep { $targets->[$_] ne "-" } ( 0 .. scalar(@$targets)-1 ) ) {
	push @{$s->[6]}, SRL::prop->new($targets->[$i], $i);
    }

    return $s; 
}



#------------------------------------------------------------


# reads the propositions of a sentence from files
# allows to store propositions as gold and/or predicted, 
#  by specifying filehandles as values in the %FILES hash
#  indexed by {GOLD,PRED} keys
# expects: each prop file: first column specifying target verbs,
#          and remaining columns specifying arguments
# returns a new sentence, containing the list of prop 
#         objects, one for each column, in gold/pred contexts
# returns undef when EOF
sub read_props {
    my ($pkg, $id, %FILES) = @_; 

    my $s; 
    my $length = undef; 

    if (exists($FILES{GOLD})) {
	my $cols = read_columns($FILES{GOLD}); 

	# end of file 
	if (!@$cols) {
	    return undef; 
	}

	$s = $pkg->new($id); 
	my $targets = shift @$cols;     
	$length = scalar(@$targets); 
	$s->set_length($length); 
	$s->load_props($s->[5], $targets, $cols); 	
    }
    if (exists($FILES{PRED})) {
	my $cols = read_columns($FILES{PRED}); 
	my $targets = shift @$cols;
	if (defined($length) and $length != scalar(@$targets)) {
	    die "ERROR : sentence $id : gold and pred sentences do not align correctly!\n"; 
	}
	$s->load_props($s->[6], $targets, $cols); 
    }

    return $s; 
}


sub load_props {
    my ($s, $where, $targets, $cols) = @_;
    
    my $i; 
    for ($i=0; $i<@$targets; $i++) {
	if ($targets->[$i] ne "-") {
	    my $prop = SRL::prop->new($targets->[$i], $i); 

	    my $col = shift @$cols;
	    if (defined($col)) {
#	    print "SE Tagging: ", join(" ", @$col), "\n"; 
		$prop->load_SE_tagging(@$col); 
	    }
	    else {
		print STDERR "WARNING : sentence ", $s->id, " : can't find column of args for prop ", $prop->verb, "!\n"; 
	    }
	    push @$where, $prop;
	}
    }
}


# writes a sentence to an output stream
# allows to specify which parts of the sentence are written
#  by giving true values to the %WHAT hash, indexed by
#  {WORDS,SYNT,GOLD,PRED} keys
# TODO : chunks and clauses are not written yet
sub write_to_stream {
    my ($s, $fh, %WHAT) = @_;

    if (!%WHAT) {
	%WHAT = ( WORDS => 1, 
		  SYNT => 1,
		  GOLD => 0,
		  PRED => 1
		  );
    }

    my @columns; 

    if ($WHAT{WORDS}) {
	my @words = map { sprintf("%-20s", $_->form) } $s->words; 
	push @columns, \@words; 
    }
    if ($WHAT{SYNT}) {
	my @pos = map { sprintf("%-6s", $_->pos) } $s->words; 
	push @columns, \@pos; 
    }
    if ($WHAT{GOLD}) {
	push @columns, $s->preds_to_columns($s->[5]); 
    }
    if ($WHAT{PRED}) {
	push @columns, $s->preds_to_columns($s->[6]); 
    }

    # finally, print columns word by word
    my $i;
    for ($i=0;$i<$s->length;$i++) {
	print $fh join(" ", map { $_->[$i] } @columns), "\n"; 
    }
    print "\n"; 


}

# turns a set of propositions (target verbs + args for each one) into a set of 
#  columns in the CoNLL-2004 Start-End format
sub preds_to_columns {
    my ($s, $Pref) = @_; 

    my @props = sort { $a->position <=> $b->position } @{$Pref}; 

    my $l = $s->length;
    my $verbs = []; 
    my @cols = ( $verbs ); 
    my $p; 

    foreach $p ( @props ) {
	defined($verbs->[$p->position]) and die "sentence->preds_to_columns: already defined verb at sentence ", $s->id, " position ", $p->position, "!\n"; 
	$verbs->[$p->position] = sprintf("%-15s", $p->verb); 

	my @tags = $p->to_SE_tagging($l); 
	push @cols, \@tags; 
    }

    # finally, define empty verb positions
    my $i;
    for ($i=0;$i<$l;$i++) {
	if (!defined($verbs->[$i])) {
	    $verbs->[$i] = sprintf("%-15s", "-"); 
	}
    }
    
    return @cols; 
}



# reads columns until blank line or EOF
# returns an array of columns (each column is a reference to an array containing the column)
# each column in the returned array should be the same size
sub read_columns {
    my $fh = shift; 

    # read columns until blank line or eof
    my @cols; 
    my $i; 
    my @line = split(" ", <$fh>); 
    while (@line) {
	for ($i=0; $i<@line; $i++) {
	    push @{$cols[$i]}, $line[$i]; 
	}
	@line = split(" ", <$fh>); 
    }
    
    return \@cols; 
}


1;






##################################################################
#
#  Package    p r o p  :  A proposition (verb + args)
#
#  January 2004
#
##################################################################

package SRL::prop;

# Constructor: creates a new prop, with empty arguments
# Parameters: verb form, position of verb
sub new {
    my ($pkg, $v, $position) = @_;

    my $p = [];
    
    $p->[0] = $v;         # the verb
    $p->[1] = $position;  # verb position
    $p->[2] = undef;      # verb sense
    $p->[3] = [];         # args, empty by default 

    return bless $p, $pkg; 
}

## Accessor/Initialization methods

# returns the verb form of the prop
sub verb {
    my $p = shift; 
    return $p->[0];
}

# returns the verb position of the verb in the prop
sub position {
    my $p = shift; 
    return $p->[1];
}

# returns the verb sense of the verb in the prop
sub sense {
    my $p = shift; 
    return $p->[2];
}

# initializes the verb sense of the verb in the prop
sub set_sense {
    my $p = shift; 
    $p->[2] = shift;
}


# returns the list of arguments of the prop
sub args {
    my $p = shift; 
    return @{$p->[3]};
}

# initializes the list of arguments of the prop
sub set_args {
    my $p = shift; 
    @{$p->[3]} = @_;
}

# adds arguments to the prop
sub add_args {
    my $p = shift; 
    push @{$p->[3]}, @_;
}

# Returns the list of phrases of the prop
# Each argument corresponds to one phrase, except for
# discontinuous arguments, where each piece forms a phrase
sub phrases {
    my $p = shift; 
    return map { $_->single ? $_ : $_->phrases} @{$p->[3]};
}


######   Methods

# Adds arguments represented in Start-End tagging
# Receives a list of Start-End tags (one per word in the sentence)
# Creates an arg object for each argument in the taggging 
#  and modifies the prop so that the arguments are part of it
# Takes into account special treatment for discontinuous arguments
sub load_SE_tagging {
    my ($prop, @tags) = @_;

    # auxiliar phrase set
    my $set = SRL::phrase_set->new(); 
    $set->load_SE_tagging(@tags); 
    
    # store args per type, to be able to continue them
    my %ARGS;
    my $a; 

    # add each phrase as an argument, with special treatment for multi-phrase arguments (A C-A C-A)
    foreach $a ( $set->phrases ) {

	# the phrase continues a started arg
	if ($a->type =~ /^C\-/) {
	    my $type = $';   # '
	    if (exists($ARGS{$type})) {
		my $pc = $a;  
		$a = $ARGS{$type}; 
		if ($a->single) {
		    # create the head phrase, considered arg until now
		    my $ph = SRL::phrase->new($a->start, $a->end, $type);
		    $a->add_phrases($ph); 
		}
		$a->add_phrases($pc); 
		$a->set_end($pc->end);
	    }
	    else {
		print STDERR "WARNING : found continuation phrase \"C-$type\" without heading phrase: turned into regular $type argument.\n";
		# turn the phrase into arg
		bless $a, "SRL::arg"; 
		$a->set_type($type); 
		push @{$prop->[3]}, $a; 
		$ARGS{$a->type} = $a; 
	    }
	}
	else {
	    # turn the phrase into arg
	    bless $a, "SRL::arg"; 
	    push @{$prop->[3]}, $a; 
	    $ARGS{$a->type} = $a; 
	}
    }

}


## discriminates the args of prop $pb wrt the args of prop $pa, returning intersection(a^b), a-b and b-a
# returns a hash reference containing three lists: 
# $out->{ok} : args in $pa and $pb
# $out->{ms} : args in $pa and not in $pb
# $out->{op} : args in $pb and not in $pa
sub discriminate_args {
    my ($pa, $pb) = @_;

    my $out; 
    @{$out->{ok}} = (); 
    @{$out->{ms}} = (); 
    @{$out->{op}} = (); 

    my $a; 
    my %ok; 

    my %ARGS; 

    foreach $a ($pa->args) {
	$ARGS{$a->start}{$a->end} = $a; 
    }

    foreach $a ($pb->args) {
	my $s = $a->start; 
	my $e = $a->end;
       	
	my $gold = $ARGS{$s}{$e}; 
	if (!defined($gold)) {
	    push @{$out->{op}}, $a;
	}
	elsif ($gold->single and $a->single) {
	    if ($gold->type eq $a->type) {
		push @{$out->{ok}}, $a;
		delete($ARGS{$s}{$e}); 
	    }
	    else {
		push @{$out->{op}}, $a;
	    }
	}
	elsif (!$gold->single and $a->single) {
	    push @{$out->{op}}, $a;
	}
	elsif ($gold->single and !$a->single) {
	    push @{$out->{op}}, $a;
	}
	else {
	    # Check phrases of arg
	    my %P; 
	    map { $P{ $_->start.".".$_->end } = $_->type } $gold->phrases;
	    my $ok = 1;
	    my @P = $a->phrases; 
	    while ($ok and @P) {
		my $p = shift @P; 
		if ($P{ $p->start.".".$p->end } ne $p->type) {
		    $ok = 0;
		}
		else {
		    delete $P{ $p->start.".".$p->end }
		}
	    }
	    if ($ok and !(values %P)) {
		push @{$out->{ok}}, $a;
		delete $ARGS{$s}{$e}
	    }
	    else {
		push @{$out->{op}}, $a;
	    }
	}
    }
    
    my ($s); 
    foreach $s ( keys %ARGS ) {
	foreach $a ( values %{$ARGS{$s}} ) {
	    push @{$out->{ms}}, $a;
	}
    }

    return $out; 
}


# Generates a Start-End tagging for the prop
# Expects the prop object, and l=length of the sentence
# Returns a list of l tags
sub to_SE_tagging {
    my $prop = shift; 
    my $l = shift; 
    my @tags = (); 

    my $p; 
    foreach $p ( $prop->phrases ) {
	my $t = $p->type; 
	if (defined($tags[$p->start])) {
	    die "prop->to_SE_tagging: Already defined tag in position ", $p->start, "! Prop phrases overlap or embed!\n";
	}
	if ($p->start != $p->end) {
	    $tags[$p->start] = sprintf("%7s", "(".$p->type)."*       ";
	    if (defined($tags[$p->end])) {
		die "prop->to_SE_tagging: Already defined tag in position ", $p->end, "! Prop phrases overlap or embed!\n";
	    }
	    $tags[$p->end] = "       *".sprintf("%-7s", $t.")"); 
	}
	else {
	    $tags[$p->start] = sprintf("%7s", "(".$t)."*".sprintf("%-7s", $t.")"); 
	}
    }
    
    my $i; 
    for ($i=0; $i<$l; $i++) {
	if (!defined($tags[$i])) {
	    $tags[$i] = "       *       ";
	}
    }

    return @tags; 
}


# generates a string representing the proposition
sub to_string {
    my $p = shift; 

    my $s = "[". $p->verb . "@" . $p->position . ": ";
    $s .= join(" ", map { $_->to_string } $p->args);
    $s .= " ]";
    
    return $s; 
}


1;


################################################################################
#
#   Package    p h r a s e _ s e t 
#
#   A set of phrases
#   Each phrase is indexed by (start,end) positions
#
#   Holds non-overlapping phrase sets. 
#   Embedding of phrases allowed and exploited in class methods
#
#   Brings useful functions on phrase sets, such as: 
#     - Load phrases from tag sequences in IOB1, IOB2, Start-End formats
#     - Retrieve a phrase given its (start,end) positions
#     - List phrases found within a given (s,e) segment
#     - Discriminate a predicted set of phrases with respect to the gold set
#
################################################################################

package SRL::phrase_set;

## $phrase_types global variable 
#  If defined, contains a hash table specifying the phrase types to be considered
#  If undefined, any phrase type is considered
my $phrase_types = undef; 
sub set_phrase_types {
    $phrase_types = {};
    my $t; 
    foreach $t ( @_ ) {
        $phrase_types->{$t} = 1; 
    } 
}

# Constructor: creates a new phrase set
# Arguments: an initial set of phrases, which are added to the set
sub new {
    my ($pkg, @P) = @_;
    my $s = [];
    @{$s->[0]} = ();     # NxN half-matrix, storing phrases
    $s->[1] = 0;         # N   (length of the sentence)
    bless $s, $pkg;

    $s->add_phrases(@P);

    return $s;
}


# Adds phrases represented in IOB2 tagging
# Receives a list of IOB2 tags (one per word in the sentence)
# Creates a phrase object for each phrase in the taggging 
#  and modifies the set so that the phrases are part of it
sub load_IOB2_tagging {
    my ($set, @tags) = @_;

    my $wid = 0;  # word id
    my $phrase = undef;  # current phrase 
    my $t;
    foreach $t (@tags) {
        if ($phrase and $t !~ /^I/) {
            $phrase->set_end($wid-1);
            $set->add_phrases($phrase);
	    $phrase = undef;
        }
        if ($t =~ /^B-/) {
            my $type = $';
            if (!defined($phrase_types) or $phrase_types->{$type}) {
		$phrase = SRL::phrase->new($wid);
		$phrase->set_type($type);
	    }
        }
        $wid++;
    }
    if ($phrase) {
        $phrase->set_end($wid-1);
        $set->add_phrases($phrase);
    }
}


# Adds phrases represented in IOB1 tagging
# Receives a list of IOB1 tags (one per word in the sentence)
# Creates a phrase object for each phrase in the taggging 
#  and modifies the set so that the phrases are part of it
sub load_IOB1_tagging {
    my ($set, @tags) = @_;

    my $wid = 0;  # word id
    my $phrase = undef;  # current phrase 
    my $t = shift @tags;
    while (defined($t)) {
	if ($t =~ /^[BI]-/) {
	    my $type = $';
            if (!defined($phrase_types) or $phrase_types->{$type}) {
		$phrase = SRL::phrase->new($wid);
		$phrase->set_type($type);
		my $tag = "I-".$type;
		$t = shift @tags;
		$wid++;
		while ($t eq $tag) {
		    $t = shift @tags;
		    $wid++;
		}
		$phrase->set_end($wid-1);
		$set->add_phrases($phrase);
	    }
	    else {
		$t = shift @tags;
		$wid++;
	    }
	}
	else {
	    $t = shift @tags;
	    $wid++;
	}
    }
}

# Adds phrases represented in Start-End tagging
# Receives a list of Start-End tags (one per word in the sentence)
# Creates a phrase object for each phrase in the taggging 
#  and modifies the set so that the phrases are part of it
sub load_SE_tagging {
    my ($set, @tags) = @_;

    my (@SP);          # started phrases
    my $wid = 0;
    my ($tag, $p); 
    foreach $tag ( @tags ) {
	while ($tag !~ /^\*/) {
	    $tag =~ /^\(([^\*\(]+)/ or die "phrase_set->load_SE_tagging: opening phrases -- bad format in $tag!\n";
	    my $type = $1;
	    $tag = $';
	    if (!defined($phrase_types) or $phrase_types->{$type}) {
		$p = SRL::phrase->new($wid);
		$p->set_type($type);
		push @SP, $p;
	    }
	}
	$tag =~ s/^\*//;
	while ($tag ne "") {
	    $tag =~ /^([^\)]+)\)/  or die "phrase_set->load_SE_tagging: closing phrases -- bad format in $tag!\n";
	    my $type = $1;
	    $tag = $';
	    if (!defined($phrase_types) or $phrase_types->{$type}) {
		$p = pop @SP;
		($type eq $p->type) or die "phrase_set->load_SE_tagging: types do not match!\n";
		$p->set_end($wid);	    
	    
		if (@SP) {
		    $SP[$#SP]->add_phrases($p);
		}
		else {
		    $set->add_phrases($p);
		}
	    }
	}
	$wid++;
    }
    (!@SP) or die "phrase_set->load_SE_tagging: some phrases are unclosed!\n";
}


# ------------------------------------------------------------

#  Adds phrases in the set, recursively (ie. internal phrases are also added)
sub add_phrases {
    my ($s, @P) = @_;
    my $ph;
    foreach $ph ( map { $_->dfs } @P ) {
	$s->[0][$ph->start][$ph->end] = $ph;
	if ($ph->end >= $s->[1]) {
	    $s->[1] = $ph->end +1;
	}
    }
}

# returns the number of phrases in the set
sub size {
    my $set = shift;

    my ($i,$j);
    my $n;
    for ($i=0; $i<@{$set->[0]}; $i++) {
	if (defined($set->[0][$i])) {
	    for ($j=$i; $j<@{$set->[0][$i]}; $j++) {
		if (defined($set->[0][$i][$j])) {
		    $n++;
		}
	    }
	}
    }
    return $n;
}

# returns the phrase starting at word position $s and ending at $e
#  or undef if it doesn't exist
sub phrase {
    my ($set, $s, $e) = @_;    
    return $set->[0][$s][$e];
}


# Returns phrases in the set, recursively in depth first search order
#  that is, if a phrase is returned, all its subphrases are also returned
# If no parameters, returns all phrases
# If a pair of positions is given ($s,$e), returns phrases included
#  within the $s and $e positions
sub phrases {
    my $set = shift;
    my ($s, $e);
    if (!@_) {
	$s = 0;
	$e = $set->[1]-1;
    }
    else {
	($s,$e) = @_;
    }
    my ($i,$j); 
    my @P = ();
    for ($i=$s;$i<=$e;$i++) {
	if (defined($set->[0][$i])) {
	    for ($j=$e;$j>=$i;$j--) {
		if (defined($set->[0][$i][$j])) {
		    push @P, $set->[0][$i][$j];
		}
	    }
	}
    }
    return @P;
}


# Returns phrases in the set, non-recursively in sequential order
#  that is, if a phrase is returned, its subphrases are not returned
# If no parameters, returns all phrases
# If a pair of positions is given ($s,$e), returns phrases included
#  within the $s and $e positions
sub top_phrases {
    my $set = shift;
    my ($s, $e);
    if (!@_) {
	$s = 0;
	$e = $set->[1]-1;
    }
    else {
	($s,$e) = @_;
    }
    my ($i,$j); 
    my @P = ();
    $i = $s;
    while ($i<=$e) {
	$j=$e;
	while ($j>=$s) {
	    if (defined($set->[0][$i][$j])) {
		push @P, $set->[0][$i][$j];
		$i=$j;
		$j=-1;
	    }
	    else {
		$j--;
	    }
	}
	$i++;
    }
    return @P;
}


# returns the phrases which contain the terminal $wid, in bottom-up order
sub ancestors {
    my ($set, $wid) = @_;

    my @A; 
    my $N = $set->[1];

    my ($s,$e); 

    for ($s = $wid; $s>=0; $s--) {
	if (defined($set->[0][$s])) {
	    for ($e = $wid; $e<$N; $e++) {
		if (defined($set->[0][$s][$e])) {
		    push @A, $set->[0][$s][$e];
		}
	    }
	}
    }
    
    return @A; 
}


# returns a TRUE value if the phrase $p ovelaps with some phrase in
#  the set; the returned value is the reference to the conflicting phrase
# returns FALSE otherwise 
sub check_overlapping { 
    my ($set, $p) = @_;

    my ($s,$e); 
    for ($s=0; $s<$p->start; $s++) {
	if (defined($set->[0][$s])) {
	    for ($e=$p->start; $e<$p->end; $e++) {
		if (defined($set->[0][$s][$e])) {
		    return $set->[0][$s][$e];
		}
	    }
	}
    }
    for ($s=$p->start+1; $s<=$p->end; $s++) {
	if (defined($set->[0][$s])) {
	    for ($e=$p->end+1; $e<$set->[1]; $e++) {
		if (defined($set->[0][$s][$e])) {
		    return $set->[0][$s][$e];
		}
	    }
	}
    }
    
    return 0; 
}


## ----------------------------------------

# Discriminates a set of phrases (s1) wrt the current set (s0), returning 
#  intersection (s0^s1), over-predicted (s1-s0) and missed (s0-s1)
# Returns a hash reference containing three lists: 
#   $out->{ok} : phrases in $s0 and $1
#   $out->{op} : phrases in $s1 and not in $0
#   $out->{ms} : phrases in $s0 and not in $1
sub discriminate {
    my ($s0, $s1) = @_; 

    my $out; 
    @{$out->{ok}} = (); 
    @{$out->{ms}} = (); 
    @{$out->{op}} = (); 

    my $ph; 
    my %ok; 

    foreach $ph ($s1->phrases) {
	my $s = $ph->start; 
	my $e = $ph->end; 
	
	my $gph = $s0->phrase($s,$e); 
	if ($gph and $gph->type eq $ph->type) {
	    # correct
	    $ok{$s}{$e} = 1; 
	    push @{$out->{ok}}, $ph;
	}
	else {
	    # overpredicted
	    push @{$out->{op}}, $ph;
	}
    }

    foreach $ph ($s0->phrases) {
	my $s = $ph->start; 
	my $e = $ph->end; 
	
	if (!$ok{$s}{$e}) {
	    # missed
	    push @{$out->{ms}}, $ph;
	}
    }
    return $out; 
}


# compares the current set (s0) to another set (s1)
# returns the number of correct, missed an over-predicted phrases
sub evaluation {
    my ($s0, $s1) = @_;

    my $o = $s0->discriminate($s1); 

    my %e; 
    $e{ok} = scalar(@{$o->{ok}}); 
    $e{op} = scalar(@{$o->{op}}); 
    $e{ms} = scalar(@{$o->{ms}}); 

    return %e;
}


# generates a string representing the phrase set, 
# for printing purposes
sub to_string {
    my $s = shift;
    return join(" ", map { $_->to_string } $s->top_phrases);
}


1;






##################################################################
#
#  Package   p h r a s e  :  a generic phrase
#
#  January 2004
#
#  This class represents generic phrases. 
#  A phrase is a sequence of contiguous words in a sentence.
#  A phrase is identified by the positions of the start/end words
#  of the sequence that the phrase spans.
#  A phrase has a type. 
#  A phrase may contain a list of internal subphrases, that is, 
#  phrases found within the phrase. Thus, a phrase object is seen
#  eventually as a hierarchical structure. 
#  
#  A syntactic base chunk is a phrase with no internal phrases.
#  A clause is a phrase which may have internal phrases
#  A proposition argument is implemented as a special class which
#  inherits from the phrase class.
#
##################################################################

package SRL::phrase; 

# Constructor: creates a new phrase
# Parameters: start position, end position and type
sub new {
    my $pkg = shift;

    my $ph = [];

    # start word index
    $ph->[0] = (@_) ? shift : undef;
    # end word index
    $ph->[1] = (@_) ? shift : undef;
    # phrase type
    $ph->[2] = (@_) ? shift : undef;
    # 
    @{$ph->[3]} = ();

    return bless $ph, $pkg; 
}

# returns the start position of the phrase
sub start {
    my $ph = shift;
    return $ph->[0];
}

# initializes the start position of the phrase
sub set_start {
    my $ph = shift;
    $ph->[0] = shift;
}

# returns the end position of the phrase
sub end {
    my $ph = shift;
    return $ph->[1];
}

# initializes the end position of the phrase
sub set_end {
    my $ph = shift;
    $ph->[1] = shift;
}

# returns the type of the phrase
sub type {
    my $ph = shift;
    return $ph->[2];
}

# initializes the type of the phrase
sub set_type {
    my $ph = shift;
    $ph->[2] = shift;
}

# returns the subphrases of the current phrase
sub phrases {
    my $ph = shift;
    return @{$ph->[3]};
}

# adds phrases as subphrases
sub add_phrases {
    my $ph = shift;
    push @{$ph->[3]}, @_;
}

# initializes the set of subphrases
sub set_phrases {
    my $ph = shift;
    @{$ph->[3]} = @_;
}


# depth first search
# returns the phrases rooted int the current phrase in dfs order 
sub dfs {
    my $ph = shift;
    return ($ph, map { $_->dfs } $ph->phrases);
}


# generates a string representing the phrase (and subphrases if arg is a TRUE value), for printing
sub to_string {
  my $ph = shift;
  my $rec = ( @_ ) ? shift : 1; 

  my $str = "(" . $ph->start . " ";

  $rec and  map { $str .= $_->to_string." " } $ph->phrases;

  $str .= $ph->end . ")";
  if (defined($ph->type)) {
      $str .= "_".$ph->type;
  }
  return $str;
}


1;

##################################################################
#
#  Package    a r g  :  An argument
#
#  January 2004
#
#  This class inherits from the class "phrase".
#  An argument is identified by start-end positions of the 
#  string spanned by the argument in the sentence.
#  An argument has a type. 
#
#  Most of the arguments consist of a single phrase; in this
#  case the argument and the phrase objects are the same.
#
#  In the special case of discontinuous arguments, the argument
#  is an "arg" object which contains a number of phrases (one
#  for each discontinuous piece). Then, the argument spans from
#  the start word of its first phrase to the end word of its last 
#  phrase. As for the composing phrases, the type of the first one
#  is the type of the argument, say A, whereas the type of the 
#  subsequent phrases is "C-A" (continuation tag). 
# 
##################################################################

package SRL::arg; 


#push @SRL::arg::ISA, 'SRL::phrase'; 
use base qw(SRL::phrase); 


# Constructor "new" inherited from SRL::phrase

# Checks whether the argument is single (returning true)
# or discontinuous (returning false)
sub single {
    my ($a) = @_;
    return scalar(@{$a->[3]}==0);
}

# Generates a string representing the argument
sub to_string {
    my $a = shift; 

    my $s = $a->type."_(" . $a->start . " ";
    map { $s .= $_->to_string." " } $a->phrases;
    $s .= $a->end . ")";

    return $s;
}


1;









##################################################################
#
#  Package   w o r d  :  a word
#
#  January 2004
#
#  A word, containing id (position in sentence), form and PoS tag
#
##################################################################

package SRL::word; 

# Constructor: creates a new word
# Parameters: id (position), form and PoS tag
sub new {
    my ($pkg, @fields) = @_;

    my $w = []; 

    $w->[0] = shift @fields;  # id (position in sentence)
    $w->[1] = shift @fields;  # form
    $w->[2] = shift @fields;  # PoS
    
    return bless $w, $pkg; 
}

# returns the id of the word
sub id {
    my $w = shift;
    return $w->[0];
}

# returns the form of the word
sub form {
    my $w = shift;
    return $w->[1];
}

# returns the PoS tag of the word
sub pos {
    my $w = shift;
    return $w->[2];
}

sub to_string {
    my $w = shift; 
    return "w@".$w->[0].":".$w->[1].":".$w->[2];
}

1;





